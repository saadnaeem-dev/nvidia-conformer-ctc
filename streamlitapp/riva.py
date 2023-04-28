import nemo.collections.asr as nemo_asr
from nemo.core.classes import IterableDataset

import torch
import torchaudio
from torch.utils.data import DataLoader

import pandas as pd
from tqdm import tqdm

import math
import numpy as np


def load_model():
    return nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_large")

class AudioChunkIterator():
    def __init__(self, samples, frame_len, sample_rate):
        self._samples = samples
        self._chunk_len = frame_len*sample_rate
        self._start = 0
        self.output=True
   
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        last = int(self._start + self._chunk_len)
        if last <= len(self._samples):
            chunk = self._samples[self._start: last]
            self._start = last
        else:
            chunk = np.zeros([int(self._chunk_len)], dtype='float32')
            samp_len = len(self._samples) - self._start
            chunk[0:samp_len] = self._samples[self._start:len(self._samples)]
            self.output = False
        return chunk

def speech_collate_fn(batch):
    """collate batch of audio sig, audio len
    Args:
        batch (FloatTensor, LongTensor):  A tuple of tuples of signal, signal lengths.
        This collate func assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    _, audio_lengths = zip(*batch)

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
   
    audio_signal= []
    for sig, sig_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        
    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    return audio_signal, audio_lengths

# simple data layer to pass audio signal
class AudioBuffersDataLayer(IterableDataset):
    def __init__(self):
        super().__init__()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._buf_count == len(self.signal) :
            raise StopIteration
        self._buf_count +=1
        return torch.as_tensor(self.signal[self._buf_count-1], dtype=torch.float32), \
               torch.as_tensor(self.signal_shape[0], dtype=torch.int64)
        
    def set_signal(self, signals):
        self.signal = signals
        self.signal_shape = self.signal[0].shape
        self._buf_count = 0

    def __len__(self):
        return 1
    
class ChunkBufferDecoder:
    def __init__(self,asr_model, stride, chunk_len_in_secs=1, buffer_len_in_secs=3):
        self.asr_model = asr_model
        self.asr_model.eval()
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(self.data_layer, batch_size=1, collate_fn=speech_collate_fn)
        self.buffers = []
        self.all_preds = []
        self.chunk_len = chunk_len_in_secs
        self.buffer_len = buffer_len_in_secs
        assert(chunk_len_in_secs<=buffer_len_in_secs)
        
        feature_stride = asr_model._cfg.preprocessor['window_stride']
        self.model_stride_in_secs = feature_stride * stride
        self.n_tokens_per_chunk = math.ceil(self.chunk_len / self.model_stride_in_secs)
        self.blank_id = len(asr_model.decoder.vocabulary)
        self.plot=False
        
    @torch.no_grad()    
    def transcribe_buffers(self, buffers, merge=True, plot=False):
        self.plot = plot
        self.buffers = buffers
        self.data_layer.set_signal(buffers[:])
        self._get_batch_preds()      
        return self.decode_final(merge)
    
    def _get_batch_preds(self):
        device = self.asr_model.device
        for batch in iter(self.data_loader):

            audio_signal, audio_signal_len = batch

            audio_signal, audio_signal_len = audio_signal.to(device), audio_signal_len.to(device)
            log_probs, encoded_len, predictions = self.asr_model(input_signal=audio_signal, input_signal_length=audio_signal_len)
            preds = torch.unbind(predictions)
            for pred in preds:
                self.all_preds.append(pred.cpu().numpy())
    
    def decode_final(self, merge=True, extra=0):
        self.unmerged = []
        self.toks_unmerged = []
        # index for the first token corresponding to a chunk of audio would be len(decoded) - 1 - delay
        delay = math.ceil((self.chunk_len + (self.buffer_len - self.chunk_len) / 2) / self.model_stride_in_secs)

        decoded_frames = []
        all_toks = []
        for pred in self.all_preds:
            ids, toks = self._greedy_decoder(pred, self.asr_model.tokenizer)
            decoded_frames.append(ids)
            all_toks.append(toks)

        for decoded in decoded_frames:
            self.unmerged += decoded[len(decoded) - 1 - delay:len(decoded) - 1 - delay + self.n_tokens_per_chunk]
        if self.plot:
            for i, tok in enumerate(all_toks):
                plt.plot(self.buffers[i])
                plt.show()
                print("\nGreedy labels collected from this buffer")
                print(tok[len(tok) - 1 - delay:len(tok) - 1 - delay + self.n_tokens_per_chunk])                
                self.toks_unmerged += tok[len(tok) - 1 - delay:len(tok) - 1 - delay + self.n_tokens_per_chunk]
            print("\nTokens collected from succesive buffers before CTC merge")
            print(self.toks_unmerged)

        if not merge:
            return self.unmerged
        return self.greedy_merge(self.unmerged)
    
    def _greedy_decoder(self, preds, tokenizer):
        s = []
        ids = []
        for i in range(preds.shape[0]):
            if preds[i] == self.blank_id:
                s.append("_")
            else:
                pred = preds[i]
                s.append(tokenizer.ids_to_tokens([pred.item()])[0])
            ids.append(preds[i])
        return ids, s
         
    def greedy_merge(self, preds):
        decoded_prediction = []
        previous = self.blank_id
        for p in preds:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(p.item())
            previous = p
        hypothesis = self.asr_model.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis


def getNemoTranscription(audio_file_path: str, model) -> str:
    """
    Generates transcription using Nemo ASR Model for specified file.
    
    Parameters:
        audio_file_path: Path to specified audio file
        
    Returns: Transcription of specified audio file.
    """
    # Load the audio and resample it to 16000 Hz
    audio,sr = torchaudio.load(audio_file_path)
    audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Reduce audio to 1 channel if it's 2 channel audio
    if len(audio.shape) == 2:
        if audio.shape[0] == 2:
            audio = torch.mean(audio,dim=0)
        elif audio.shape[0] == 1:
            audio = audio.squeeze(0)
            
    # Convert audio to numpy array
    audio = audio.numpy()
    
    # Set up buffer and chunk lengths, stride, and sample rate
    context_len_in_secs = 1
    chunk_len_in_secs = 15            
    stride = 4
    sample_rate = 16000
    chunk_len = sample_rate * chunk_len_in_secs
    buffer_len_in_secs = chunk_len_in_secs + 2 * context_len_in_secs
    buffer_len = sample_rate * buffer_len_in_secs
    sampbuffer = np.zeros([buffer_len], dtype=np.float32)
    
    # Read chunks of audio
    chunk_reader = AudioChunkIterator(audio, chunk_len_in_secs, sample_rate)
    buffer_list = []
    for chunk in chunk_reader:
        sampbuffer[:-chunk_len] = sampbuffer[chunk_len:]
        sampbuffer[-chunk_len:] = chunk
        buffer_list.append(np.array(sampbuffer))
    
    # Transcribe the audio chunks
    asr_decoder = ChunkBufferDecoder(model, stride=stride, chunk_len_in_secs=chunk_len_in_secs, buffer_len_in_secs=buffer_len_in_secs)
    transcription = asr_decoder.transcribe_buffers(buffer_list, plot=False)
    
    return transcription

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

import pandas as pd
import numpy as np
from pydub import AudioSegment
from pydub import silence
from tqdm.notebook import tqdm


# In[2]:


MODEL_NAME = "openai/whisper-large-v2"

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
processor = WhisperProcessor.from_pretrained(MODEL_NAME)

model.eval()
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")

whisper_pipeline = pipeline("automatic-speech-recognition",
                            model=model,
                            tokenizer=processor.tokenizer,
                            feature_extractor=processor.feature_extractor,
                            chunk_length_s=30,
                            device=torch.device('cpu'),
                            max_new_tokens=1024)


# In[3]:


def getWhisperTranscript(audio) -> str:
    # Convert audio tensor to a numpy array
    samples = audio.get_array_of_samples()
    audio = np.array(samples)
    
    # Generate the transcription using the whisper_pipeline
    transcription = whisper_pipeline(audio)['text']
    
    # Return the transcription
    return transcription


# In[9]:


def merge_conversation(list1, list2, column=3):
    merged_list = []
    i = j = 0
    n1, n2 = len(list1), len(list2)
    while i < n1 and j < n2:
        if list1[i][column] <= list2[j][column]:
            merged_list.append(list1[i])
            i += 1
        else:
            merged_list.append(list2[j])
            j += 1
    merged_list.extend(list1[i:])
    merged_list.extend(list2[j:])
    return merged_list


# In[10]:


def split_on_silence(audio, silence_detect_fn, label):
    silences = silence_detect_fn(audio)
    chunks = []
    start = 0
    for i, (start_silence, end_silence) in enumerate(silences):
        transcript = ""
        end = start_silence * 1000
        mid =  (start + end) / 2
        chunk = audio[start:end]
        if chunk.duration_seconds > 0.5:
            transcript = getWhisperTranscript(chunk)        
        chunks.append((label, start/1000, end/1000, mid, chunk, transcript))
        start = end_silence * 1000
    end = audio.duration_seconds * 1000
    mid =  (start + end) / 2
    chunk = audio[start:end]
    if chunk.duration_seconds > 0.5:
        transcript = getWhisperTranscript(chunk)   
    chunks.append((label, start/1000, end/1000, mid, chunk, transcript))
    return chunks

def extract_silences(audio, dBFS=None):
    if dBFS is None:
        dBFS = audio.dBFS
    sil = silence.detect_silence(audio, min_silence_len=1000, silence_thresh=dBFS-16)
    return [(start/1000, stop/1000) for start, stop in sil]


# In[14]:


def dialog_transcription(agent_file, customer_file):
    # Read Audio files using pydub
    agent_audio = AudioSegment.from_file(agent_file)
    customer_audio = AudioSegment.from_file(customer_file)
        
    #Resampling to 16k
    agent_audio = agent_audio.set_frame_rate(16000)
    customer_audio = customer_audio.set_frame_rate(16000)
    
    #split audios based on silences then merge and order
    agent_chunks = split_on_silence(agent_audio, extract_silences, "Agent")
    customer_chunks = split_on_silence(customer_audio, extract_silences, "Customer")
    dialog =  merge_conversation(agent_chunks, customer_chunks)
    
    # Concatenate transcript of the chunks with their timestamps
    result = ""
    for label, start, end, mid, chunk, transcript in dialog:
        if chunk.duration_seconds > 0.5: 
            #print("[{:<4.2f}, {:<4.2f}] {:<8}: {}".format(start, end, label, transcript))
            chunk_str = "[{:<4.2f}, {:<4.2f}] {:<8} :{}\n".format(start, end, label, transcript)
            result += chunk_str
    
    return result


# In[15]:


transcript = dialog_transcription("agent.wav", "customer.wav")
print(transcript)


# In[16]:


transcript = dialog_transcription("MegaAudio/d3nn9h1jsqje8dv9v0pq_1_TFS_400059.wav", "MegaAudio/d3nn9h1jsqje8dv9v0pq_101011.wav")
print(transcript)


# In[17]:


transcript = dialog_transcription("MegaAudio/1snipln0j6obdfsbeqhv_400059.wav", "MegaAudio/1snipln0j6obdfsbeqhv_1_TFS_101011.wav")
print(transcript)






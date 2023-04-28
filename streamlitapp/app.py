import os
import riva
import streamlit as st
from pydub import AudioSegment
import whisper

st.set_page_config(
    page_title="ASR",
    page_icon="musical_note",
    layout="wide",
    initial_sidebar_state="auto",
)

audio_tags = {'comments': 'Converted using pydub!'}

upload_path = "uploads/"
download_path = "downloads/"
transcript_path = "transcripts/"

# @st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def to_mp3(audio_file, output_audio_file, upload_path, download_path):
    ## Converting Different Audio Formats To MP3 ##
    if audio_file.name.split('.')[-1].lower()=="wav":
        audio_data = AudioSegment.from_wav(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="mp3":
        audio_data = AudioSegment.from_mp3(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="ogg":
        audio_data = AudioSegment.from_ogg(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="wma":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"wma")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="aac":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"aac")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="flac":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"flac")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="flv":
        audio_data = AudioSegment.from_flv(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="mp4":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"mp4")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)
    return output_audio_file

def process_audio_riva(filename, model_type):
    model = riva.load_model()
    result = riva.getNemoTranscription(filename, model)
    return result

def process_audio_whisper(filename, model_type):
    model = whisper.load_model(model_type)
    decode_options = dict(language="en", without_timestamps=True)
    transcribe_options = dict(task="transcribe", **decode_options)
    result = model.transcribe(filename, **transcribe_options)
    return result["text"]

def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file),"w") as f:
        f.write(transcript_data)

st.title("🗣 Automatic Speech Recognition Tool WHISPER/RIVA")
st.info('Supported audio formats - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV')
uploaded_file = st.file_uploader("Upload audio file", type=["wav","mp3","ogg","wma","aac","flac","mp4","flv"])

audio_file = None

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())
    with st.spinner(f"Processing Audio ... 💫"):
        output_audio_file = uploaded_file.name.split('.')[0] + '.mp3'
        output_audio_file = to_mp3(uploaded_file, output_audio_file, upload_path, download_path)
        audio_file = open(os.path.join(download_path,output_audio_file), 'rb')
        audio_bytes = audio_file.read()
    print("Opening ",audio_file)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("Play uploaded audio file")
        st.audio(audio_bytes)
    with col2:
        model_type = st.radio("Please choose your model", ('NVIDIA RIVA', 'WHISPER'))
    with col3:
        whisper_model_type = st.radio("Please choose your model type", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))

    if st.button("Generate Transcript"):
        with st.spinner(f"Generating Transcript... 💫"):
            if model_type == 'NVIDIA RIVA':
	            transcript = process_audio_riva(str(os.path.abspath(os.path.join(download_path,output_audio_file))), whisper_model_type.lower())
            elif model_type == 'WHISPER':
	            transcript = process_audio_whisper(str(os.path.abspath(os.path.join(download_path,output_audio_file))), whisper_model_type.lower())

            output_txt_file = str(output_audio_file.split('.')[0]+".txt")

            save_transcript(transcript, output_txt_file)
            output_file = open(os.path.join(transcript_path,output_txt_file),"r")
            output_file_data = output_file.read()

        st.markdown("---")
        st.markdown("<strong>Transcript output:</strong>", unsafe_allow_html=True)
        st.markdown(transcript)

        if st.download_button(
                             label="Download Transcript 📝",
                             data=output_file_data,
                             file_name=output_txt_file,
                             mime='text/plain'
                         ):
            st.balloons()
            st.success('✅ Download Successful !!')

else:
    st.warning('⚠ Please upload your audio file')

st.markdown("<br><hr><center>ASR Test Application</center><hr></br>", unsafe_allow_html=True)



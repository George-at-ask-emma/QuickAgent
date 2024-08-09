import asyncio
from dotenv import load_dotenv
import os
import time
import openai
from pyaudio import PyAudio, paFloat32

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)
        start_time = time.time()
        response = self.conversation.invoke({"text": text})
        end_time = time.time()
        self.memory.chat_memory.add_ai_message(response['text'])
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']

class TextToSpeech:
    def __init__(self, interrupt_event):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.interrupt_event = interrupt_event

    async def speak(self, text):
        print("Converting text to speech...")
        try:
            player = PyAudio().open(format=paFloat32, channels=1, rate=24000, output=True)
            with openai.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="shimmer",
                response_format="pcm",
                input=text,
            ) as stream:
                for chunk in stream.iter_bytes(chunk_size=1024):
                    if self.interrupt_event.is_set():
                        print("Speech interrupted.")
                        break
                    player.write(chunk)
                    await asyncio.sleep(0)
            print("Finished playing audio response.")
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
        finally:
            player.close()

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.transcript_collector = TranscriptCollector()
        self.interrupt_event = asyncio.Event()
        self.new_input_event = asyncio.Event()

    async def get_transcript(self):
        try:
            print("Initializing Deepgram client...")
            config = DeepgramClientOptions(options={"keepalive": "true"})
            deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
            if not deepgram_api_key:
                raise ValueError("DEEPGRAM_API_KEY not found in environment variables")
            deepgram: DeepgramClient = DeepgramClient(deepgram_api_key, config)
            dg_connection = deepgram.listen.asynclive.v("1")
            print("Deepgram client initialized. Listening...")

            def on_message(result):
                print("Received transcription result")
                sentence = result.channel.alternatives[0].transcript
                
                if not result.speech_final:
                    print(f"Partial transcription: {sentence}")
                    self.transcript_collector.add_part(sentence)
                    self.interrupt_event.set()
                else:
                    self.transcript_collector.add_part(sentence)
                    full_sentence = self.transcript_collector.get_full_transcript()
                    if len(full_sentence.strip()) > 0:
                        full_sentence = full_sentence.strip()
                        print(f"Human: {full_sentence}")
                        self.transcription_response = full_sentence
                        self.new_input_event.set()
                        self.transcript_collector.reset()

            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

            options = LiveOptions(
                model="nova-2",
                punctuate=True,
                language="en-US",
                encoding="linear16",
                channels=1,
                sample_rate=16000,
                endpointing=600,
                smart_format=True,
            )

            print("Starting Deepgram connection...")
            await dg_connection.start(options)

            print("Initializing microphone...")
            microphone = Microphone(dg_connection.send)
            microphone.start()
            print("Microphone started. Listening for input...")

            while True:
                await asyncio.sleep(1)
                print("Waiting for speech input...")

        except Exception as e:
            print(f"Error in get_transcript: {str(e)}")

    async def main(self):
        print("Starting main loop...")
        transcription_task = asyncio.create_task(self.get_transcript())
        tts = TextToSpeech(self.interrupt_event)

        try:
            while True:
                print("Waiting for new input...")
                await self.new_input_event.wait()
                print("New input received!")
                self.new_input_event.clear()
                self.interrupt_event.clear()

                if "goodbye" in self.transcription_response.lower():
                    print("Goodbye detected. Exiting...")
                    break

                print(f"Processing input: {self.transcription_response}")
                llm_response = self.llm.process(self.transcription_response)
                print(f"LLM response: {llm_response}")
                await tts.speak(llm_response)

                self.transcription_response = ""
        finally:
            print("Cancelling transcription task...")
            transcription_task.cancel()
            try:
                await transcription_task
            except asyncio.CancelledError:
                pass
            print("Transcription task cancelled. Exiting...")

if __name__ == "__main__":
    print("Initializing ConversationManager...")
    manager = ConversationManager()
    print("Running main asyncio loop...")
    asyncio.run(manager.main())
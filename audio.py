import pyaudio
import wave

#Changement du repertoire de travaille
os.chdir("/home/ray974/Learning/")

def enregistrement(nomFichier, genre):


    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5

    #Ouveture des fichiers pour ajouter les noms des fichiers reçus
    Hfiles = open("VoiceRecord/homme/Hfiles",'a')
    Ffiles = open("VoiceRecord/femme/Ffiles",'a')

    if (genre == "h"):
        #Destination du fichier wav
        WAVE_OUTPUT_FILENAME = "VoiceRecord/homme/test_" + nomFichier + ".wav"
        #Ecriture du nom dans le fichier correspondant pour effectuer la fft a posteriori
        fileNameH = "test_" + nomFichier
        Hfiles.write(fileNameH)

    elif (genre == "f"):
        #Destination du fichier wav
        WAVE_OUTPUT_FILENAME = "VoiceRecord/femme/test_" + nomFichier + ".wav"
        #Ecriture du nom dans le fichier correspondant pour effectuer la fft a posteriori
        fileNameF = "test_" + nomFichier
        Ffiles.write(fileNameF)

    else :
        print("ATTENTION. Si vous ête un homme, tapez 'h', une femme tapez 'f'. Sinon, désolé.")

    Hfiles.close()
    Ffiles.close()

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Commencez à parler, vous avez " + str(RECORD_SECONDS) + " secondes.")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Enregistrement terminé, merci.")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


enregistrement("test_W","h")
import csv

def analyseFile(fname):
    with open(fname, 'rt') as f:
        reader = csv.reader(f)
        nbRow = 0
        for col in reader:
            if (nbRow >= 10):
                break
            if (col[5]=="male"):
                print("male")
            elif (col[5]=="female"):
                print("female")
            else:
                print('*')
            nbRow += 1




analyseFile("/media/ray974/common-voice/cv-valid-dev.csv")

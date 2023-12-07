1. Librariile necesare pentru rularea solutiei:

numpy==1.26.2
opencv_python==4.8.1.78
python==3.11.5


2. Cum ar trebui rulat codul si unde putem gasi output-ul:

Task 1, 2, 3:

script: main.py
fisiere_auxiliare: folderul templates (aici se afla imaginile cu ajutorul carora realizam template matching pentru fiecare piesa de domino)

Instructiuni de rulare:
In main regasim randurile urmatoare, pe liniile 258, 259:

(258)	folder_imagini = 'antrenare'
(259)	folder_rezultate = 'fisiere_solutie'

(a)  input: variabilei folder_imagini ii atribuim numele folderului cu imaginile din care trebuie sa extragem informatia (format imagini: .jpg)
(b) output: variabilei folder_rezultate ii atribuim numele folderului in care vor fi fisierele text cu rezultatele (format fisier: .txt)
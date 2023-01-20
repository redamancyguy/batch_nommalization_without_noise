import pdfplumber


def find_string_from_pdf(file_path, string):
    result = []
    with pdfplumber.open(file_path) as pdf:
        for i in range(len(pdf.pages)):
            page_text = pdf.pages[i].extract_text().lower().replace('\n', ' ').replace('  ', ' ')
            # page_text = pdf.pages[i].extract_text().lower()
            # print(page_text)
            if type(string) is str:
                if page_text.__contains__(string):
                    result.append(string)
            elif type(string) is list:
                for _ in string:
                    if page_text.__contains__(_):
                        result.append(_)
            else:
                raise Exception("type error !")
        return list(set(result))


father_path = '/Users/sunwenli/Downloads/paper-filter'
import os


def file_name(file_path):
    for root, dirs, files in os.walk(file_path):
        return files


files = file_name(father_path)
for i in files:
    if not i.__contains__('pdf') or i[0] == '.':
        continue
    try:
        result = find_string_from_pdf(father_path + '/' + i, ['sgd', 'gradient', 'descent'])
        if len(result) > 0:
            print(i, 'result : ', result)
    except Exception as e:
        print(e)





    #  and find_string_from_pdf(father_path + '/' + i, ['lsm', 'nosql', 'unstructured']):
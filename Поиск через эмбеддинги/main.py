from PdfReader import load_pdfs
from GigaChatWrapper import GigaChatWrapper

from pandas import read_csv, DataFrame


#configure_logger('pdf_reader.log')


PAPERS = [
    "14.3-bit Extended Counting ADC with Built-in Binning Function for Medical X-Ray Imagers.pdf",
    "A 9-V_Lux-s 5000-Framess 512x512 CMOS Sensor.pdf",
    "A Flexible 14-Bit Column-Parallel ADC Concept for Application in Wafer-Scale X-ray CMOS Imagers.pdf",
]


QUESTIONS_WITH_SOURCE = {
    "О чём статья?" : PAPERS[0],
    "С каким техпроцессом был изготовлен прототип?" : PAPERS[0], # 0.35 мкм

    "О чём статья?" : PAPERS[1],
    "Опиши архитектуру сенсора" : PAPERS[1],
    "С какой максимальной частотой сенсор корректно работает?" : PAPERS[1], # 5000 fps

    "О чём статья?" : PAPERS[2],
    "Какая рабочая частота смены кадров устройства?" : PAPERS[2], # 100 fps
    "Какая глубина конверсии?" : PAPERS[2], # 14 bit
    "Какое напряжение питания устройства?" : PAPERS[2], # 3.5V
}




def main() -> None:
    # update text database
    database = 'data.csv'
    folder_name = 'C:\\Desktop\\bmnk\\projects\\BorisKaryshev___LLM_in_doc_search\\Поиск через эмбеддинги\\Статьи'

    database = load_pdfs(folder_name, database, num_of_jobs=1)
    
    database.to_csv('data.csv', index=False)

    chat = GigaChatWrapper(database)

    for question in QUESTIONS_WITH_SOURCE.keys():
        res = chat.ask_question(question)
        print(
            f'Question: {question}\n'
            f'Answer: {res}\n'
        )


if __name__ == "__main__":
    main()

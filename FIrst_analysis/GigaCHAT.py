from langchain.chat_models.gigachat import GigaChat
from langchain.prompts.chat import ChatPromptTemplate

CREDENTIALS = "ZjkyMWJhMmUtMTA3OC00NzBmLTk1NTctZjkwZTFlM2EwMjEzOjBkZmNkMzcxLWM3Y2EtNGRmOS1hYzlmLWZjMjBmNGNlMDJlMg=="

QUESTIONS = [
    "Подбери транзистор с наибольшим коэффициентом усиления",
    "Подбери транзистор с максимальным коэффициентом усиления и с напряжением коллектор-эмиттер больше 40 В",
    "Подбери транзистор с минимальным током коллектора и коэффициентом усиления более 150",
    "Подбери транзистор с максимальным током коллектора и коэффициентом усиления более 300",
    "Подбери транзистор с максимальным коэффициентом усиления и током коллектора не меньше 0.15 А",
    "Подбери транзистор с максимальным коэффициентом усиления и током коллектора не меньше 150 мА",
    "Что больше 151 мА или 0,15 А?",
]

TEMPLATE = (
    "Ты эксперт по компонентной базе микроэлектроники. "
    "Твоя задача помочь выбрать оптимальный компонент согласно пользовательскому запросу. "
    "Выбранный тобой компонент обязательно должен соответствовать ВСЕМ ТРЕБОВАНИЯМ ПОЛЬЗОВАТЕЛЯ!."
    "Отвечай кратко.\n"
    "Обязательно укажи значения характеристик, упомянутых пользователем\n"
    "Например: [BC598] : [Коэффициент усиления 30, ток коллектора 1 А]\n"    
    "Далее приведены характеристики различных электронных компонентов \"{datasheet}\""
)

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TEMPLATE),
    ("human", "{text}"),
])

def main(data_filename: str) -> None:
    with open(data_filename, 'r', encoding='utf-8') as file: 
        datasheet = file.read()

    chat = GigaChat(credentials=CREDENTIALS, verify_ssl_certs=False)
    for question in QUESTIONS:
        prompt = CHAT_PROMPT.format_messages(
            datasheet=datasheet,
            text=question
        )
        res = chat.invoke(prompt)
        print(f"Question: {question}\nBot: {res.content}\n")

        
if __name__ == "__main__":
    print("Даташит подётся на вход в табличном виде")
    main("table_data.txt")
    print("Даташит подётся на вход в текстовом виде")
    main("text_data.txt")

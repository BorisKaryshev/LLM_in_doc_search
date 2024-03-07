from pandas import read_csv, DataFrame
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import GigaChatEmbeddings
from langchain.chat_models.gigachat import GigaChat

CREDENTIALS = "ZjkyMWJhMmUtMTA3OC00NzBmLTk1NTctZjkwZTFlM2EwMjEzOjBkZmNkMzcxLWM3Y2EtNGRmOS1hYzlmLWZjMjBmNGNlMDJlMg=="

def create_embeddings(data: DataFrame | str, max_tokens: int = 512) -> DataFrame:
    data_path = None
    if isinstance(data, str):
        data_path = data
        data = read_csv(data)
        
    names_with_text = {}

    for idx in data.index:
        print(data['name'][idx])
        name = data['name'][idx]
        text = data['text'][idx]

        if name not in names_with_text:
            names_with_text[name] = [text]
        else:
            names_with_text[name].append(text)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                        chunk_size=max_tokens, chunk_overlap=5, separator=' ',
                    )
    for (name, text) in names_with_text.items():
        names_with_text[name] = text_splitter.split_text(''.join(text))
        print(
            'Chunk lengths:\n'
            + ' '.join([str(len(chunk.split(' '))) for chunk in names_with_text[name]])
        )
    
    
    chat_model = GigaChat(credentials=CREDENTIALS)
    copy = names_with_text
    names_with_text = {}
    for (name, text) in copy.items():
        names_with_text[name] = [chunk for chunk in text if chat_model.get_num_tokens(chunk) <= max_tokens]
    

    result_data = {
        'name' : [],
        'text' : [],
        'embedding' : [],
        'chunk' : [],
    }
    
    embedder = GigaChatEmbeddings(credentials=CREDENTIALS, verify_ssl_certs=False)
    for (name, text) in names_with_text.items():
        for (index, chunk) in enumerate(text):
            
            result_data['name'].append(name)
            result_data['text'].append(chunk)
            result_data['chunk'].append(index)
            
            embedding = embedder.embed_documents(texts=[chunk])[0]
            result_data['embedding'].append(embedding)

    data = DataFrame(result_data)
    if data_path is not None:
        data.to_csv(data_path)

    return data
        
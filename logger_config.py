import logging
import sys

def setup_logger():
    """
    Configura o logger principal para o projeto.

    Esta função define o formato das mensagens de log, o nível de detalhe
    e os handlers que direcionam o output para o console e para um arquivo.
    """
    # Define o formato da mensagem de log
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configura o logger raiz.
    # Nível INFO: Captura mensagens de INFO, WARNING, ERROR, CRITICAL.
    # Para ver mensagens de DEBUG, mude para logging.DEBUG.
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler("app.log"),  # Envia logs para o arquivo 'app.log'
            logging.StreamHandler(sys.stdout) # Envia logs para o console
        ]
    )

    # Impede que loggers de bibliotecas de terceiros (muito "barulhentas")
    # poluam o log principal. Ajuste conforme necessário.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
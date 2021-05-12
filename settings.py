from pydantic import BaseSettings

ENV_FILE = '.env'
ENV_FILE_ENCODING = 'utf-8'


class AppSettings(BaseSettings):
    """Global configuration class that loads information from the environment.
    """

    class Config:
        env_file = ENV_FILE
        env_file_encoding = ENV_FILE_ENCODING

    device: str = 'cpu'
    train: bool = True
    epochs: int = 1000
    lr: float = 3e-4

    image_size: int = 512
    patch_size: int = 32
    train_batch_size: int = 4

    model_weights_folder: str


settings = AppSettings()

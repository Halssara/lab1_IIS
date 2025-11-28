import joblib
import pandas as pd

class FastAPIHandler:
    def __init__(self, model_path: str):
        """
        Инициализация класса:
        Загружаем весь pipeline, включая все этапы предобработки и модель.
        """
        # Загружаем обученную модель (pipeline)
        self.model = joblib.load(model_path)
        print(f"Модель загружена из {model_path}")


    def predict(self, features: pd.DataFrame) -> float:
        """
        Получает список признаков (в теле POST-запроса),
        обрабатывает их и возвращает предсказание.
        """
        prediction = self.model.predict(features)[0]
        return int(prediction)

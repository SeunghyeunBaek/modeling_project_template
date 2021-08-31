from models.customnet import CustomNet

def get_model(model_name: str, model_args: dict):
    """모델 클래스 변수 설정
    Args:
        model_str (str): 모델 클래스명
    Note:
        model_str 이 모델 클래스명으로 정의돼야함
        `model` 변수에 모델 클래스에 해당하는 인스턴스 할당
    """

    if model_name == 'Linear':
        return CustomNet(**model_args)

if __name__ == '__main__':
    pass
# Anti-spoofing project
## Overview
Репозиторий для обучения RawNet2 на ASVspoof2019 датасете. Итоговое качество на тестовой выборке: 4.4% EER.

## Installation guide
Устанавливаются нужные библиотеки. При обучении использовался датасет с
Kaggle, поэтому, если хотите обучить/запустить модель локально надо
в файле hw_as/trainer/trainer.py указать в качестве переменной 
asv_scores_file путь до файла ASVspoof2019.LA.asv.eval.gi.trl.scores.txt
из датасета, а также изменить root в конфиге.
```shell
cd HSE_DLA_AS
pip install -r ./requirements.txt
```

## Training model
Флаги в квадратных скобках не использовались при обучении, но
при желании их можно использовать. 

-r - путь до чекпойнта, если хотите продолжить обучение модели

```shell
cd HSE_DLA_AS
python3 train.py -c hw_as/configs/config.json
                 [-r default_test_model/checkpoint.pth]
```

## Testing
Код для тестирования скачанного чекпойнта. По дефолту берутся
аудио из папки test_data, но можно указать путь до своей
папки с аудиозаписями. После того, как код выполнится,
в файл out.txt запишутся вероятности того, что аудио фейковое.

Значения флагов

-r - путь до чекпойнта, где находится модель

-c - путь до конфига, если потребутся, что-то дополнительное

-t - путь до директории с тестовыми аудио, которые надо подать модели

-o - путь до файла, куда будет записываться результат

```shell
cd HSE_DLA_AS
python3  -r pretrained_models/model.pth \
         [-c your config if needed]
         [-t path to test dir with .wav/.flac files]
         [-o out_file]
```

## Author
Юрий Максюта, БПМИ203
import os
from collections import defaultdict

def collect_unique_extensions(path='/mnt/nfs/Learning/'):
    extensions = defaultdict(int)

    for root, dirs, files in os.walk(path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext and len(ext) <= 5:  # Учитываем точку в расширении, поэтому длина <= 5
                ext = ext.lower()  # Приводим расширение к нижнему регистру
                extensions[ext] += 1

    return list(extensions.keys())

# Пример использования
unique_extensions = collect_unique_extensions()
print("Уникальные расширения файлов:", unique_extensions)

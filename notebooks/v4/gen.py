import os

saved_models = """
gs://neural-guidance-tensorflow/export/0315_series5a_sr100_l30_aTrue_tng-tpu-05/1552771408/
gs://neural-guidance-tensorflow/export/0315_series5a_sr100_l30_aTrue_tng-tpu-pre-11/1552775337/
gs://neural-guidance-tensorflow/export/0315_series5a_sr100_l30_aTrue_tng-tpu-pre-17/1552777164/
gs://neural-guidance-tensorflow/export/0315_series5a_sr100_l30_aTrue_tng-tpu-pre-23/1552818144/
gs://neural-guidance-tensorflow/export/0315_series5a_sr100_l30_aTrue_tng-tpu-pre-29/1552784319/
gs://neural-guidance-tensorflow/export/0315_series5a_sr30_l30_aTrue_tng-tpu-00/1552780821/
gs://neural-guidance-tensorflow/export/0315_series5a_sr30_l30_aTrue_tng-tpu-06/1552782486/
gs://neural-guidance-tensorflow/export/0315_series5a_sr30_l30_aTrue_tng-tpu-pre-12/1552786382/
gs://neural-guidance-tensorflow/export/0315_series5a_sr30_l30_aTrue_tng-tpu-pre-18/1552788060/
gs://neural-guidance-tensorflow/export/0315_series5a_sr30_l30_aTrue_tng-tpu-pre-24/1552792412/
gs://neural-guidance-tensorflow/export/0315_series5a_sr50_l40_aTrue_tng-tpu-02/1552773431/
gs://neural-guidance-tensorflow/export/0315_series5a_sr50_l40_aTrue_tng-tpu-08/1552773406/
gs://neural-guidance-tensorflow/export/0315_series5a_sr50_l40_aTrue_tng-tpu-pre-14/1552775461/
gs://neural-guidance-tensorflow/export/0315_series5a_sr50_l40_aTrue_tng-tpu-pre-20/1552775302/
gs://neural-guidance-tensorflow/export/0315_series5a_sr50_l40_aTrue_tng-tpu-pre-26/1552775320/
gs://neural-guidance-tensorflow/export/0315_series5a_sr70_l30_aTrue_tng-tpu-04/1552751496/
gs://neural-guidance-tensorflow/export/0315_series5a_sr70_l30_aTrue_tng-tpu-pre-10/1552753367/
gs://neural-guidance-tensorflow/export/0315_series5a_sr70_l30_aTrue_tng-tpu-pre-16/1552757983/
gs://neural-guidance-tensorflow/export/0315_series5a_sr70_l30_aTrue_tng-tpu-pre-22/1552761897/
gs://neural-guidance-tensorflow/export/0315_series5a_sr70_l30_aTrue_tng-tpu-pre-28/1552763161/
""".strip().split()


with open("template.ipynb") as f:
  template = f.read()
placeholder = "gs://neural-guidance-tensorflow/export/0315_series5a_sr30_l30_aTrue_tng-tpu-pre-24/1552792412/"


for x in saved_models:
  name = x.split('/')[-3]
  print(name)
  with open(name + ".ipynb", "w") as f:
    f.write(template.replace(placeholder, x))

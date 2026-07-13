export CUDA_VISIBLE_DEVICES=5
celery -A django_project worker -l info -c 4

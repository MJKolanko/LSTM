# LSTM
Okej, daje tutaj drobny opis tego co się dzieje:
1. Dane - mamy folder data do którego plik download_data.py ściaga zipa z librispeech i go rozpakowuje do data/clean. Jest tez folder data/noise, do którego wrzuciłam ściągnięte ręcznie 300 różnych zakłóceń z WHAM. Plik preprare_clean_librispeech.py dodatkowo konwertuje .flac (który pochodzi z datasetu librispeech oryginalnie) na .wav. WHAM jest już w .wav od początku.
2. Dataset - prepare_data.py miksuje speakerów i daje im noise.
3. Trening denoiser - train_denoiser.py jest po 30 epokach treningu, ale używam tam ConvTasNeta dla lepszych wyników - są w sumie spoko. 
4. Infer denoise - Przelatuje przez cały dataset i go odszumia. Dodałam tam też czas ile to trwa, średnio na 6s mixa to było około 0.02 s.
5. Separator - na razie trenuje go na czystych danych, żeby zobaczyć czy działa, a potem zrobie checkpoint dla danych zaszumionych dla porównania. Jeszcze nie wiem jak będzie - jestem na 3 epoce (40 epok x 8 minut, także well) i wyglądają wyniki obiecująco:
██████████████████████████████████████████████████████████████████████████████ 1000/1000 [08:09<00:00,  2.04it/s]
Epoch 1: train=-0.204688 | val=-2.563453
Saved separator -> sep_model.pt
██████████████████████████████████████████████████████████████████████████████ 1000/1000 [07:32<00:00,  2.21it/s]
Epoch 2: train=-3.585769 | val=-4.195527
Saved separator -> sep_model.pt
██████████████████████████████████████████████████████████████████████████████ 1000/1000 [07:32<00:00,  2.21it/s]
Epoch 3: train=-5.066674 | val=-5.424089
Saved separator -> sep_model.pt
7. Tu zrobie test separacji, nie testowałam tego kodu, ale przykładowy jest w infer_separator.py.
8. Na koniec ewaluacja tego wszystkiego za pomocą evaluate.py i zobaczymy jak to będzie.

UWAGA - mixuje 3 speakerów - dlaczego? Wiem, że miałam próbować na więcej, ALE:
* Jeśli chcemy fajne wyniki korzystam z modeli do tego stworzonych (takich ostro porządnych) i je dotrenowuje na mniejszym datasecie dla 3 speakerów. Żeby zrobić 4-5 trzeba uczyć model od zera praktycznie, co nie jest najlepszym wyjściem, bo musiałby mieć dużo dużo teningu (chat mówi, że w granicy 10000 mixów na 100 epokach) dlatego też poszłam na kompromis i mamy 3 (jeśli zadziała). 
* Denoiser jest niezależny!! On nie bierze pod uwagę ilu jest speakerów tylko wykrywa noise i go usuwa, także tam nie trzeba tego precyzować i można robić co się święcie wymarzy.

Także tak moja droga widownio, denoiser działa - i to działa całkiem nieźle muszę przyznać. Lepiej niż jakakolwiek implementacja całego innego *eghem*, które mamy opisane w załozeniach projektowych. Z wynikami separatora przyjdę jutro i podziele się nimi przed lub po tym jak dostanę mental breakdance.

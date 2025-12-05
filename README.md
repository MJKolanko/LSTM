# LSTM
Okej, daje tutaj drobny opis tego co się dzieje:
1. Dane - mamy folder data do którego plik download_data.py ściaga zipa z librispeech i go rozpakowuje do data/clean. Jest tez folder data/noise, do którego wrzuciłam ściągnięte ręcznie 300 różnych zakłóceń z WHAM. Plik preprare_clean_librispeech.py dodatkowo konwertuje .flac (który pochodzi z datasetu librispeech oryginalnie) na .wav. WHAM jest już w .wav od początku.
2. Dataset - prepare_data.py miksuje speakerów i daje im noise
3. Trening denoiser - train_denoiser.py jest po 30 epokach treningu, ale używam tam ConvTasNeta dla lepszych wyników - są w sumie spoko. 
4. Infer denoise - Przelatuje przez cały dataset i go odszumia. Dodałam tam też czas ile to trwa, średnio na 6s mixa to było około 0.02 s.
5. Separator - na razie trenuje go na czystych danych, żeby zobaczyć czy działa, a potem zrobie checkpoint dla danych zaszumionych dla porównania. Jeszcze nie wiem jak będzie - jestem na 3 epoce (40 epok x 8 minut, także well) i wyglądają wyniki obiecująco:
██████████████████████████████████████████████████████████████████████████████ 1000/1000 [08:09<00:00,  2.04it/s]
Epoch 1: train=-0.204688 | val=-2.563453
Saved separator -> sep_model.pt
██████████████████████████████████████████████████████████████████████████████ 1000/1000 [07:32<00:00,  2.21it/s]
Epoch 2: train=-3.585769 | val=-4.195527
Saved separator -> sep_model.pt
6. Tu zrobie test separacji (jeszcze nie ma kodu)
7. Na koniec ewaluacja tego wszystkiego i zobaczymy jak to będzie

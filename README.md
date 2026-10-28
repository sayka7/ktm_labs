# KTM laboratoriyalari — Sun'iy intellektga asoslangan kompyuter tizimlari muhandisligi

Fan sayti: https://aiscan.airi.uz/ktm/ (sillabus, ma'ruza matnlari, virtual laboratoriya). Bu repozitoriyda virtual laboratoriyaning barcha bosqich kodlari va qo'shimcha topshiriqlar — kompyuteringizga ko'chirib olib bajarish uchun.

## Tez boshlash

```bash
git clone https://github.com/sayka7/ktm_labs.git
cd ktm_labs
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd A1 && python3 s1_*.py          # yoki ./run_all.sh
```

Docker bilan (saytdagi virtual laboratoriya bilan bir xil muhit):

```bash
docker build -t ktm-lab .
docker run --rm -it -v $PWD:/work -w /work ktm-lab python3 A1/s1_*.py
```

Talablar: Python 3.10+, ~2 GB disk (torch CPU). GPU shart emas — barcha kodlar CPU da ishlaydi; GPU bo'lsa izohlardagi ko'rsatmalar bo'yicha o'zgartirish mumkin.

## Laboratoriyalar

| Kod | Mavzu | Bosqichlar | Topshiriqlar |
|---|---|---|---|
| [A1](A1/) | Sun'iy intellektga asoslangan kompyuter tizimlari arxitekturasini tahlil qilish va loyihalash | 7 | 10 |
| [A2](A2/) | GPU va sun'iy intellekt tezlatkichlaridan foydalangan holda parallel hisoblashni tashkil etish | 7 | 10 |
| [A3](A3/) | Sun'iy intellekt tizimlarini optimallashtirish va unumdorligini baholash | 6 | 10 |
| [A4](A4/) | Edge AI va Cloud AI platformalarida sun'iy intellekt tizimlarini joylashtirish | 6 | 10 |
| [A5](A5/) | Taqsimlangan sun'iy intellekt tizimlarini ishlab chiqish | 7 | 10 |
| [A6](A6/) | Sun'iy intellekt tizimlarining xavfsizligi va ishonchliligini tahlil qilish | 7 | 10 |
| [A7](A7/) | Yakuniy kompleks loyihani shakllantirish: ma'lumot -> model -> API -> monitoring -> adolatlilik | 7 | 10 |

Har papkada `README.md` (bosqichlar, kutilgan natija, 10 topshiriq), `VARIANTS.md` (20 ta mustaqil variant), `s1_*.py … sN_*.py` (mustaqil dasturlar) va `run_all.sh` bor.

## Baholash

Har laboratoriyada 10 qo'shimcha topshiriq: 3 oson (5 ball), 4 o'rta (10 ball), 3 qiyin (15 ball) = 100 ball. Hisobot: kod + natijalar + qisqa xulosa.

© 2026-2027 Sun'iy intellektga asoslangan kompyuter tizimlari muhandisligi. O'quv maqsadida.

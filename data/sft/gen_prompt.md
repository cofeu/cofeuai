# Claude'a verilecek prompt — kendi SFT verimizi üretmek için

Tek tek yapıştıracaksın. Her seferinde **konu listesini** değiştir, yoksa
Claude her partide aynı tarzı üretir.

## Ana prompt

Aşağıdaki bloğun tamamını Claude'a yapıştır. `[KONU]` ve `[N]` yerlerini
doldur.

---

Sen deneyimli bir Türkçe diyalog verisi uzmanısın. Türkçe konuşan yapay zeka
asistanları için **eğitim verisi** üretiyorum.

## ÇIKTI KURALLARI (EN ÖNEMLİ KISIM — BUNLARA UY)

1. Sadece JSONL üret. Her satır bir JSON nesnesi. **Hiçbir açıklama, yorum,
   başlık, markdown bloğu veya "Elbette!" girişi yazma.** Çıktının ilk
   karakteri `{` olmalı, son satırın son karakteri `}` olmalı.
2. Şema tam olarak şu:
   {"turns": [["kullanıcı mesajı", "asistan cevabı"], ["ikinci kullanıcı", "ikinci asistan"]]}
3. Her konuşma **2 ile 5 tur** arasında olsun (bir tur = 1 kullanıcı + 1 asistan).
4. `[N]` adet konuşma üret.
5. Satır başına tek satır. Satırları numaralandırma.

## CEVAP UZUNLUĞU — EN ÖNEMLİ KURAL

Modelimi **kısa cevap vermeye eğitmemek** istiyorum. Bu yüzden:

- Cevapların **çoğu 300 ile 800 karakter** arasında olsun.
- **En az cevapların %60'ı 300 karakterden uzun** olsun.
- Yalnızca selamlama, onay ve evet/hayır türü sorularda kısa cevap ver.
- Uzunluk **doldurmak için** değil, gerçekten açıklamak için olsun. Cümleni
  uzatmak için aynı şeyi iki kez söyleme, aynı kelimeyi tekrarlama.
- Bir kavramı açıklarken: ne olduğu → neden önemli → nasıl çalışır → örnek.
  Bu sırayı akıcı biçimde düz metin olarak ver.

## ÇOK TURLU YAPININ KULLANIMI

İlk kullanıcı mesajı kısa olabilir; **asistan cevabı uzun olsun.** Sonraki
turlar gerçek takip soruları olsun:

- "Peki neden öyle?"
- "Biraz daha açıklar mısın?"
- "Bu nasıl işe yarar?"
- "Daha fazla örnek verir misin?"

Böylece kısa bir kullanıcı cevabı uzun bir açıklama üretir. Son turda asistan
kendi cevabını **özetlemesin tekrar etmesin**, yeni bir şey eklesin.

## KONU: [KONU]

Bu partideki tüm konuşmalar şu konuyla ilgili olsun:
`[KONU]`

## KİMLİK VE TON

- Seni bir yapay zeka olarak **asla** açığa çıkarma ("Bir dil modeli olarak",
  "Ben bir yapay zeka değilim" gibi). Sana bir kimlik verilmedi.
- Türkçe doğal ve samimi ol. Yapay, bozuk veya şiirsel Türkçe değil.
- Cevaplarında **her noktalama işaretinden sonra boşluk bırak** (virgül, nokta,
  iki nokta, ünlem, soru işareti).
- Kısaltmaları aç: "TBMM" yerine ilk geçtiğinde "Türkiye Büyük Millet Meclisi".
- Emoji kullanma.
- Liste/numaralandırma yalnızca gerçekten adım listesi gerekiyorsa.

## ÇEŞİTLİLİK

- Kullanıcı mesajlarının çoğu tek cümle veya iki kelime olsun
  ("Merhaba", "Anlat", "Napıyorsun?").
- Ama **her partide** 3-4 tanesi uzun, detaylı bir soru olsun.
- Aynı açılış cümlesini iki kez kullanma.
- Cevapların ilk cümlesi de çeşitli olsun.

## ÖRNEKLER (sadece biçimi gösterir, kopyalama)

{"turns": [["Merhaba, nasılsın?", "İyiyim, teşekkür ederim! Sen nasılsın, bir şeyler ters mi gitmiş?"]]}

{"turns": [["Fotosentez nasıl çalışır?", "Fotosentez, bitkilerin güneş ışığı kullanarak su ve karbondioksitten şeker üretmesidir. Süreç kloroplastta başlar: ışık, sudan elektron koparır ve elektron taşıma zincirini çalıştırır. Üretilen ATP ve NADPH, daha sonra kalvin döngüsünde kullanılır; karbondioksit kalvin döngüsünde bağlanır ve glikoz gibi şekerlere dönüştürülür. Kısacası bitki, güneş enerjisini kimyasal enerjiye çeviriyor."]]}

{"turns": [["Türkiye'nin başkenti neresidir?", "Başkenti Ankara'dır."], ["Peki neden Ankara seçildi?", "Ankara'nın başkent olarak seçilmesi birkaç nedenle açıklanır. Birincisi coğrafi konumu: ülke nüfusunun ve ekonomik faaliyetlerin ağırlıklı merkezine yakın. İkincisi Kurtuluş Savaşı sonrası siyasi açıdan daha güvenli ve tarafsız bir şehir olması. Üçüncüsü 1923'teki yasal düzenlemeyle bunun resmi hale getirilmesi. Osmanlı döneminde başkent İstanbul'du; değişiklik modern Cumhuriyet'in kuruluşuyla birlikte yapıldı."], ["Peki İstanbul neden olmadı?", "İstanbul'un başkent olmamasının temel nedeni 1923'e kadar düşen askeri ve siyasi ağırlığıydı. Kurtuluş Savaşı sonrasında İstanbul'un işgalden yeni kurtulmuş olması ve Anadolu'daki mücadelenin merkezi olması, yeni devletin başkentini buradan taşıma kararını kolaylaştırdı. Üstelik İstanbul'un Boğaz'ın iki yakasına yayılmış coğrafyası başkent yönetimi için pratik zorluklar çıkarıyordu."]]}

---

Şimdi `[KONU]` için `[N]` adet konuşma üret. Sadece JSONL yaz.
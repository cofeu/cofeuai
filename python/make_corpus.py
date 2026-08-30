"""Daha büyük bir Türkçe corpus oluşturur.

Mevcut sample_corpus.txt'yi genişletmek için ek hikayeler ve metinler üretir.
Gerçek kullanımda buraya kendi metinlerinizi ekleyebilirsiniz.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample_corpus.txt"

STORIES = [
    # Hikaye 1: Mevcut hikaye (çocuk ve gezgin)
    """Bir zamanlar uzak bir köyde yaşayan küçük bir çocuk vardı. Çocuk her sabah erkenden kalkar, tarlalara gider ve ailesine yardım ederdi. Köyün insanları onu çok severdi çünkü o her zaman güler yüzlü ve yardımseverdi.

Günlerden bir gün, köye yabancı bir gezgin geldi. Gezgin, uzak diyarlardan geldiğini ve birçok hikaye bildiğini söyledi. Çocuk, gezginin anlattığı hikayeleri büyük bir merakla dinledi. Gezgin ona denizlerin ötesindeki ülkeleri, yüksek dağları ve derin ormanları anlattı.

Çocuk o gece uyuyamadı. Aklında hep gezginin anlattığı maceralar vardı. Ertesi sabah çocuk, ailesine veda edip dünyayı keşfetmeye karar verdi. Annesi ona bir çanta yiyecek, babası ise eski bir harita verdi.

Çocuk yola çıktı. Önce büyük bir ormana girdi. Ormanda kuşlar şarkı söylüyor, ağaçlar rüzgarla dans ediyordu. Çocuk ormanın derinliklerine doğru yürüdü ve bir nehirle karşılaştı. Nehrin üzerinde eski bir köprü vardı.

Köprüyü geçerken çocuk, suyun içinde parlayan bir şey gördü. Eğilip baktığında, suyun dibinde altın bir anahtar olduğunu fark etti. Anahtarı almak için elini suya uzattı ama su çok soğuktu.

Tam o sırada yaşlı bir balıkçı yanına geldi. Balıkçı, çocuğa yardım etti ve anahtarı sudan çıkardı. Çocuk balıkçıya teşekkür etti ve yoluna devam etti.

Yolculuğu boyunca çocuk birçok yeni arkadaş edindi. Birlikte dağları aştılar, çölleri geçtiler ve sonunda büyük bir şehre ulaştılar. Şehirde insanlar çocuğu sıcak bir şekilde karşıladı.

Çocuk, dünyanın ne kadar büyük ve güzel olduğunu anladı. Ama en çok da evini ve ailesini özledi. Bir gün, öğrendiği her şeyi ailesine anlatmak için köyüne geri dönmeye karar verdi.

Köyüne döndüğünde herkes onu büyük bir sevinçle karşıladı. Çocuk, gezginin anlattığı hikayeleri ve kendi maceralarını köy halkına anlattı. O günden sonra köyde herkes, dünyayı keşfetmenin ne kadar önemli olduğunu konuşmaya başladı.

Ve çocuk, hayatı boyunca öğrenmeye ve keşfetmeye devam etti. Çünkü biliyordu ki, dünya keşfedilmeyi bekleyen sonsuz bir hazineydi.""",

    # Hikaye 2: Bilge baykuş
    """Derin bir ormanın en yüksek ağacında yaşlı bir baykuş yaşardı. Bu baykuş, ormandaki tüm hayvanların sorularına cevap veren bilge bir kuştu. Her sabah ormanın hayvanları onun ağacının altında toplanır, ona sorular sorardı.

Bir gün küçük bir tavşan, baykuşun yanına geldi. Tavşan çok üzgündü çünkü arkadaşlarıyla oynarken hep kaybediyordu. Baykuş tavşana baktı ve gülümsedi.

Kaybetmek önemli değil, dedi baykuş. Önemli olan her seferinde biraz daha iyi olmaya çalışmaktır. Tavşan bu sözleri duyunca çok sevindi ve o günden sonra her gün biraz daha çalıştı.

Aradan aylar geçti. Tavşan artık ormanın en hızlı koşucusu olmuştu. Ama o hala her gün baykuşun ağacına gelir, yeni şeyler öğrenirdi. Çünkü biliyordu ki, öğrenmenin sonu yoktur.

Baykuş da tavşanın bu azmini görünce çok mutlu oldu. Ormandaki diğer hayvanlara tavşanı örnek gösterdi. O günden sonra ormandaki tüm hayvanlar, her gün yeni bir şey öğrenmeye çalıştı.""",

    # Hikaye 3: Yıldız toplayıcısı
    """Gökyüzünün en karanlık köşesinde küçük bir yıldız toplayıcısı yaşardı. O, her gece gökyüzüne bakar ve düşen yıldızları toplardı. Topladığı yıldızları küçük bir kavanozda saklardı.

Bir gece, yıldız toplayıcısı çok parlak bir yıldızın düştüğünü gördü. Hemen koştu ve yıldızı yakalamak için ellerini uzattı. Ama yıldız o kadar parlaktı ki, gözlerini kamaştırdı.

Yıldız toplayıcısı yıldıza yaklaştı ve onun aslında bir dilek yıldızı olduğunu anladı. Dilek yıldızları çok nadir bulunurdu ve onları bulan kişi bir dilek dileyebilirdi.

Yıldız toplayıcısı uzun süre düşündü. Ne dilemeliydi? Daha fazla yıldız mı? Daha büyük bir kavanoz mu? Sonra aklına harika bir fikir geldi.

Ben, dedi yıldız toplayıcısı, tüm dünyadaki çocukların mutlu olmasını diliyorum. Yıldız parladı ve gökyüzüne yükseldi. O gece, dünyadaki tüm çocuklar güzel rüyalar gördü.

Yıldız toplayıcısı o günden sonra yıldız toplamaya devam etti. Ama artık onları kavanozda saklamıyor, gökyüzüne geri bırakıyordu. Çünkü biliyordu ki, yıldızlar en çok gökyüzünde parlarken güzeldir.""",

    # Hikaye 4: Ressam ve renkler
    """Küçük bir kasabada yaşayan genç bir ressam vardı. Ressam, resim yapmayı çok severdi ama bir sorunu vardı: renkleri karıştırmayı bilmiyordu. Her resminde renkler birbirine karışır, ortaya garip görüntüler çıkardı.

Bir gün ressam, kasabanın en yaşlı ressamını ziyaret etti. Yaşlı ressam ona renklerin sırrını anlattı. Her rengin bir hikayesi vardır, dedi. Mavi denizin, yeşil ormanın, sarı güneşin hikayesini anlatır.

Genç ressam bu sözleri duyunca çok şaşırdı. O güne kadar renkleri sadece boya olarak görmüştü. Ama şimdi anlıyordu ki, her renk bir duyguyu, bir anıyı temsil ediyordu.

Ressam eve döndü ve yeniden resim yapmaya başladı. Bu sefer renkleri karıştırmak yerine, her rengi kendi hikayesiyle kullandı. Ortaya çıkan resimler o kadar güzeldi ki, kasaba halkı hayran kaldı.

Genç ressam artık kasabanın en sevilen ressamı olmuştu. Ama o hala her gün yeni renkler keşfetmeye, yeni hikayeler öğrenmeye devam ediyordu. Çünkü biliyordu ki, sanatın da öğrenmenin de sonu yoktur.""",

    # Hikaye 5: Küçük mucit
    """Küçük bir çocuk, odasında sürekli yeni şeyler icat etmeye çalışırdı. Bazen bir uçan makine, bazen bir zaman makinesi tasarlardı. Ama icatlarının çoğu çalışmazdı.

Çocuğun babası bir gün ona dedi ki: Başarısızlık, başarının ilk adımıdır. Her başarısız deneme, sana neyin yanlış olduğunu öğretir.

Çocuk bu sözleri hiç unutmadı. Her başarısız icadından sonra, neyin yanlış gittiğini düşündü ve bir sonraki denemesinde düzeltmeye çalıştı.

Yıllar geçti. Çocuk büyüdü ve gerçek bir mucit oldu. İcat ettiği makineler insanların hayatını kolaylaştırdı. Ama o hala her başarısızlığı bir öğrenme fırsatı olarak görüyordu.

Bir gün genç mucitlere konuşma yaptı. Onlara dedi ki: Asla pes etmeyin. Çünkü en büyük icatlar, en çok başarısızlıktan sonra ortaya çıkar.""",
]


def main():
    text = "\n\n".join(STORIES)
    OUT.write_text(text, encoding="utf-8")
    print(f"Corpus yazıldı: {OUT}")
    print(f"Karakter sayısı: {len(text)}")
    print(f"Kelime sayısı (yaklaşık): {len(text.split())}")


if __name__ == "__main__":
    main()
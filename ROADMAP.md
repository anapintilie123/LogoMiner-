# Arhitectura Pipeline-ului de Scraping

Am început cu un **script secvențial** simplu, dar procesarea devenea lentă și gestionarea excepțiilor (redirect-uri, blocaje de tip bot detection) mult prea complicată. Am trecut la un **model asincron** cu `asyncio` și thread pooling, permițând lansarea simultană a multiplelor request-uri HTTP. Astfel, am putut trata mai ușor redirecționări (HTTP->HTTPS, www/non-www) și am introdus antete custom pentru a semăna cu un browser real.

## Preprocesarea Imaginilor

- **Problema**: aveam logo-uri în formate variate (SVG, ICO, PNG). Unele erau transparente, altele foarte mari.
- **Soluție**: convertirea la **format unitar** (`RGBA` + resize la 128×128). SVG-urile le transform cu `cairosvg`, iar ICO-urile mari le deschid în PIL și le pun pe fundal alb dacă aveau transparență.

## Feature Extraction

- **Inițial** am încercat doar DCT combinat cu histograme de culoare, dar weighting-ul (ex. 70% DCT, 30% histogram) nu a dat rezultate consistente.  
- **Apoi** am renunțat la histogramă, însă logo-urile cu culori similare și forme diferite erau grupate greșit.  
- **Acum** extrag și HOG pentru forme/contururi, plus pHash pentru duplicate. Rezultatul e un vector bogat (DCT, HOG, histogram) normalizat cu `StandardScaler`.

## Clustering Agglomerative

- **Ce n-a mers**:
  - **K-Means**: trebuia să aleg un număr fix de clustere din start, iar eu nu-l știam.
  - **DBSCAN**: am încercat, dar parametrii (eps/min_samples) erau dificili, iar datele formau structuri neregulate.
- **De ce Agglomerative**:
  - Pot folosi `distance_threshold` pentru a „tăia” dendrograma și obține un număr adaptiv de clustere.
  - Dendrogramele se vizualizează intuitiv.  
- **Am testat** Ward și Average linkage; Ward păstrează clustere mai compacte, Average e util când datele sunt mai difuze.


**Fișiere debug**:
  - Loguri despre redirect-uri, excepții.  
  - Un fișier de **histograme** (ex. `data/hist_debug.txt`) și fișiere `.txt` pentru valorile DCT/HOG/pHash, ca să urmăresc evoluția și să depanez rezultate suspecte.
  - Dendrograme salvate (de ex. `dendrogram_pca.png`, `clusters_dendrogram.png`) – util pentru a vedea cum se unesc logo-urile.
- **CSV** cu `(filename, domain, cluster_id)` și **foldere** per cluster – pot verifica manual dacă imaginile din același cluster se 
aseamănă vizual.

**Rezultate**
- **Interpretare**:  
  - Dacă mai multe logo-uri cu culori similare apar în același cluster, e semn că histograma color și DCT-ul au captat un pattern comun.  
  - Dacă un logo ar fi trebuit să fie “apropiat” dar e totuși separat, se pot inspecta fișierele de debug pentru a verifica HOG sau pHash.  
- **Concluzie**: Pipeline-ul actual separă coerent imagini aproape identice și grupează logo-urile în funcție de frecvențe, culoare și contur, iar structura ierarhică a clusterelor confirmă că datele pot fi tăiate la praguri diferite pentru o granularitate mai fină or mai larga.



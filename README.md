

# Estymacja dynamicznej macierzy przepływu w aglomeracji miejskiej

## Spis treści
* [Cel projektu](#cel-projektu)
* [Podstawowe pojęcia](#podstawowe-pojęcia)
* [Technologie](#technologie)
* [Konfiguracja oprogramowania](#konfiguracja-oprogramowania)
* [Uruchomienie oprogramowania](#uruchomienie-oprogramowania)
* [Funcjonalności](#funcjonalności)
* [TODO](#todo)


## Cel projektu
Celem projektu było napisanie oprogramowania obliczającego dynamiczną macierz przepływu na podstawie natężeń ruchu.

## Podstawowe pojęcia

Miasto było traktowane jak **graf**, w którym skrzyżowania były węzłami, a drogi krawędziami skierowanymi. **Natężenie ruchu** było definiowane jako ilość pojazdów, które przejechały z danego węzła źródłowego do węzła docelowego (sąsiadującego z węzłem źródłowym). **Przepływ** był definiowany jako ilość pojazdów rozpoczynających podróż z danego węzła źródłowego do danego węzła docelowego w danym przedziale czasowym. 

**Macierz przepływu** informuje o przepływach pomiędzy dowolną parą węzłów w danym przedziale czasowym. **Dynamiczna macierz przepływu** jest ciągiem macierzy przepływu dla kolejnych przedziałów czasowych.

## Technologie
* Python 3
* TensorFlow 1.15.0
* symulator SUMO

## Konfiguracja oprogramowania
Konfiguracja oprogramowania wymaga następujących czynności: utworzenie mapy dla symulatora SUMO, określenie bazowego ciągu macierzy przepływu oraz dostosowania parametrów w pliku konfiguracyjnym [config.py](https://github.com/robert-czwartosz/estymacja-dod/blob/main/config.py).

### Utworzenie mapy dla symulatora SUMO
Mapa składa się z sieci ([map.net.xml](https://github.com/robert-czwartosz/estymacja-dod/blob/main/sumo/map.net.xml)), definicji stref TAZ ([map.taz.xml](https://github.com/robert-czwartosz/estymacja-dod/blob/main/sumo/map.taz.xml)) oraz detektorów natężenia ruchu([edges.txt](https://github.com/robert-czwartosz/estymacja-dod/blob/main/sumo/detectors/edges.txt)). Sieć definiuje węzły i połączenia między nimi. Strefa TAZ(Traffic Assignmeng Zone) składa się z krawędzi, które są podzielone na źródłowe i docelowe. Z krawędzi źródłowych (source) wyjeżdżają nowe pojazdy. W krawędziach docelowych(sink) kończy się trasa pojazdów i te pojazdy "znikają". Detektory pozwalają zmierzyć natężenie ruchu na wybranych krawędziach. W pliku [edges.txt](https://github.com/robert-czwartosz/estymacja-dod/blob/main/sumo/detectors/edges.txt) znajduje się lista krawędzi, na których powinien odbywać się pomiar.

Mapę można utworzyć na dwa sposoby. Pierwszym z nich jest podanie współrzędnych węzłów i połączeń pomiędzy nimi. Drugim sposobem jest wyeksportowanie pliku map.osm z OpenStreetMap(https://www.openstreetmap.org).

Zaletą pierwszego z nich jest możliwość stworzenia dowolnego połączenia węzłów oraz brak konieczności ręcznego definiowania stref TAZ i detektorów. Jednak ten sposób ma swoje ograniczenia: wszystkie drogi są dwukierunkowe(2 pasy w każdym kierunku) z ograniczeniem prędkości do 50km/h oraz wszystkie skrzyżowania są kierowane poprzez sygnalizację świetlną. Ominięcie tych ograniczeń jest możliwe tylko poprzez modyfikację kodu programu [createNet.py](https://github.com/robert-czwartosz/estymacja-dod/blob/main/sumo/createNet.py).

Zaletą drugiego sposobu(OpenStreetMap) jest możliwość dokładnego odwzorowania dowolnego miasta przy minimalnym nakładzie pracy poświęconej na konfigurację. Wadą tego sposobu mogą być nadmiarowe połączenia, które umożliwiają poruszanie się pojazdów trasami niezgodnymi z założeniami badań.

#### Sposób 1: utworzenie własnej mapy
Najpierw należy utworzyć plik **map_net.txt** w katalogu [/sumo](https://github.com/robert-czwartosz/estymacja-dod/blob/main/sumo/); przykładowa zawartość:

	~ 	Init node 	Term node	;
		1	2	;
		1	3	;
		2	1	;
		2	6	;
		3	1	;
	.
	.
	.
		24	23	;


W pliku zawarte są możliwe połączenia pomiędzy węzłami.

Następnie należy utworzyć plik **map_node.txt** w katalogu [/sumo](https://github.com/robert-czwartosz/estymacja-dod/blob/main/sumo/); przykładowa zawartość:

	Node	X	Y	;
	1	50000	510000	;
	2	320000	510000	;
	3	50000	440000	;
	4	130000	440000	;
	5	220000	440000	;
	6	320000	440000	;
	7	420000	380000	;
	8	320000	380000	;
	9	220000	380000	;
	10	220000	320000	;
	11	130000	320000	;
	12	50000	320000	;
	13	50000	50000	;
	14	130000	190000	;
	15	220000	190000	;
	16	320000	320000	;
	17	320000	260000	;
	18	420000	320000	;
	19	320000	190000	;
	20	320000	50000	;
	21	220000	50000	;
	22	220000	130000	;
	23	130000	130000	;
	24	130000	50000	;

W pliku zawarte położenia węzłów. W pierwszej kolumnie znajduje się nr węzła. W drugiej i trzeciej kolumnie znajdują się współrzędne X i Y węzłów.

Aby wygenerować pliki **map.net.xml** oraz **map.taz.xml** w katalogu [/sumo](https://github.com/robert-czwartosz/estymacja-dod/blob/main/sumo/), należy uruchomić następujące polecenia:

	python createNet.py
	netconvert --node-files map.nod.xml --edge-files map.edg.xml -t map.type.xml -o map.net.xml**

#### Sposób 2: skorzystanie z OpenStreetMap

1. Pobierz mapę z OpenStreetMap(https://www.openstreetmap.org) i przenieś plik .map do folderu [/sumo](https://github.com/robert-czwartosz/estymacja-dod/blob/main/sumo/)
2. Uruchom następujące polecenia:

	netconvert --osm-files map.osm -o map.net.xml
	netconvert -s map.net.xml --remove-edges.by-type highway.bridleway,highway.bus_guideway,highway.cycleway,highway.footway,highway.ford,highway.path,highway.pedestrian,highway.raceway,highway.service,highway.stairs,highway.step,highway.steps,railway.highspeed,railway.light_rail,railway.preserved,railway.rail,railway.subway,railway.tram,highway.living_street --remove-edges.isolated true -o map.net.xml
3. Za pomocą programu netedit(polecenie netedit map.net.xml) usuń z sieci zbędne węzły i krawędzie.
4. Stwórz plik map.taz.xml w folderze[/sumo](https://github.com/robert-czwartosz/estymacja-dod/blob/main/sumo/) zawierające definicje badanych węzłów sieci.

Każda strefa TAZ jest definiowana jako zbiór krawędzi(edges). Krawędzie można podzielić na źródłowe i docelowe. Z krawędzi źródłowych (source) wyjeżdżają nowe pojazdy. W krawędziach docelowych(sink) kończy się trasa pojazdów i te pojazdy "znikają".
Składnia pliku map.taz.xml:
```
<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
	<taz id="Jana_pawla_Nord_10">
	  <tazSource id="114017941#1" weight="1.0"/>
	  <tazSink id="286864439" weight="1.0"/>
	</taz>
	<taz id="Jana_pawla_Sud_19">
	  <tazSource id="286864436#1" weight="1.0"/>
	  <tazSink id="634946213" weight="1.0"/>
	</taz>
	.
	.
	.
</additional>
```

Każdy znacznik TAZ składa się z listy identyfikatorów krawędzi separowanych spacjami. Identyfikator krawędzi można odczytać z programu NETEDIT w polu id poprzez kliknięcie na daną krawędź. Pod polem id jest również pole name, które pomaga w określeniu nazwy ulicy na jakiej znajduje się dana krawędź. Jeśli to pole jest puste, to można posłużyć się mapą Google w celu ustalenia nazwy ulicy.

5. Utwórz plik edges.txt w folderze [/sumo/detectors](https://github.com/robert-czwartosz/estymacja-dod/blob/main/sumo/detectors) zawierającego krawędzie, na których będą zliczane pojazdy.

Przykładowa zawartość:

	Source 15 1 114017938 
	Sink 1 15 114017937#0 
	Sink 1 10 286864439 
	Source 10 1 114017941#1 
	Sink 1 19 634946213 
	Source 19 1 286864436#1 
	T 1 71 303103884 
	.
	.
	.
	T 71 1 286864434#5 

W pierwszej kolumnie znajduje się nazwa typu krawędzi (Source - początek trasy, Sink - zakończenie trasy, T - krawędź przejezdna). W kolumnie 2 i 3 znajduje się identyfikator(według rysunku przedstawionego na początku rozdziału) węzła początkowego i końcowego. W czwartej kolumnie znajduje się identyfikator krawędzi, który można odczytać w programie NETEDIT. Każda linia powinna być zakończona spacją.

#### Sposób 3: stworzenie/edytowanie mapy w programie NETEDIT
Po zastosowaniu sposobu 2, może się okazać że sieć transportowa(map.net.xml) wymaga edycji. Wówczas należy użyć programu NETEDIT, aby usunąć, dodać lub zmodyfikować pewne elementy pliku map.net.xml.
Aby edytować stworzoną sieć, wystarczy uruchomić polecenie: netedit map.net.xml.

### Określenie bazowego ciągu macierzy przepływu
Aby uruchomić symulację SUMO, należy podać mapę oraz ciąg macierzy przepływu.
Ciąg macierzy przepływu jest losowo generowany na podstawie bazowego ciągu macierzy przepływu.
Macierz przepływów(lub ciąg macierzy) może być zmierzona np. poprzez śledzenie pojazdów po numerach rejestracyjnych.
Są dwa sposoby na uzyskanie bazowego ciągu macierzy przepływu: 
1. Mając zmierzoną macierz przepływu, jej wartości są dzielone przez liczbę przedziałów czasowych i powielenie jej.
2. Zmierzony ciąg macierzy przepływu, jest jednocześnie bazowym ciągiem macierzy przepływu.
#### Sposób 1
Utwórz plik OD.txt z macierzą przepływu opisującej ruch w badanym okresie czasu, skłądającego się z mniejszych przedziałów czasowych; przykładowa zawartość:

	source\\dest,10,15,19,690,699,710,719,760,769,910,919,980,989,1449,2225
	10,0,1301.4,625.1,5.9,34.1,75.0,269.0,38.2,79.5,77.7,116.4,263.5,321.9,1.6,38.0
	15,1044.6,0,1181.9,144.0,98.4,302.9,753.3,95.6,179.8,308.0,369.3,693.8,789.1,0.0,196.7
	19,706.7,1626.1,0,41.0,16.1,137.2,408.0,11.9,45.0,140.8,187.7,388.5,459.8,8.1,80.3
	690,61.2,177.8,14.0,0,138.8,11.5,14.1,138.6,257.8,446.2,533.8,143.0,188.0,0.8,294.4
	699,55.6,113.8,52.9,124.9,0,25.9,45.6,78.5,156.8,301.2,369.1,82.6,116.8,24.1,193.2
	710,142.4,706.4,178.5,6.1,48.3,0,219.1,21.4,1.8,64.4,95.6,214.2,261.2,10.0,32.0
	719,43.3,328.7,59.2,23.3,35.9,15.1,0,25.1,36.2,16.4,32.4,85.8,110.2,5.7,4.1
	760,2.9,194.3,117.7,227.5,157.2,4.0,16.2,0,2.1,73.3,108.3,160.1,210.0,19.7,37.9
	769,63.9,158.9,3.9,184.5,125.4,5.5,8.4,33.5,0,56.4,86.6,127.6,170.4,14.4,27.6
	910,156.7,589.7,188.5,766.3,601.6,76.8,219.2,130.3,227.4,0,2563.7,612.8,696.9,302.7,1798.4
	919,96.0,483.6,120.8,618.6,475.8,42.9,153.9,81.1,152.6,2220.1,0,484.0,563.3,269.8,1645.4
	980,91.6,557.8,119.3,60.0,26.3,36.7,157.7,21.2,68.2,199.3,262.9,0,611.0,2.5,117.4
	989,115.5,616.3,146.7,80.1,42.2,49.2,185.5,37.4,94.5,233.2,300.1,580.8,0,33.4,141.0
	1449,17.4,21.9,13.9,19.2,4.0,10.2,37.3,20.0,1.0,373.0,404.6,3.0,17.9,0,1488.7
	2225,68.4,371.7,87.3,489.8,374.6,30.0,114.9,60.8,116.8,1835.7,1951.1,377.1,441.0,1236.7,0
Bazowa macierz przepływów powinna zawierać średnie przepływy pomiędzy poszczególnymi węzłami.

Macierz nie wymaga dzielenia przez ilość przedziałów czasowych, ponieważ to zostanie wykonane w programie generateDataFromOD.py.
#### Sposób 2
Utwórz plik ODpairs.txt z parami źródło-cel (OD); przykładowa zawartość:

	1O,2D,
	1O,3D,
	1O,4D,
	1O,5D,
	1O,6D,
	1O,7D,
	.
	.
	.
	24O,23D,

Każdy wiersz zawiera jedną parę OD. Na początku jest nr węzła źródłowego + litera 'O', a po przecinku jest nr węzła docelowego + litera 'D'. Każda linia jest zakończona przecinkiem.

Utwórz plik DOD.txt z bazowym ciągiem macierzy przepływu w głównym katalogu; przykładowa zawartość:

	1	0.5	0	0.666666667	10	0.666666667	12.5	0.666666667	15	0.666666667	7.5	;
	2	0.5	0	0.666666667	10	0.666666667	12.5	0.666666667	15	0.666666667	7.5	;
	3	0.5	0	0.666666667	50	0.666666667	62.5	0.666666667	75	0.666666667	37.5	;
	4	0.5	0	0.666666667	20	0.666666667	25	0.666666667	30	0.666666667	15	;
	5	0.5	0	0.666666667	30	0.666666667	37.5	0.666666667	45	0.666666667	22.5	;
	.
	.
	.
	528	0.5	0	0.666666667	70	0.666666667	87.5	0.666666667	105	0.666666667	52.5	;

W pierwszej kolumnie znajduje się numer pary źródło-cel (OD), będący numerem linii zawierającej daną parę w pliku ODpairs.txt. Druga kolumna zawiera czasy trwania przepływów, które znajdują się w trzeciej kolumnie. Kolejne kolumny podają informacje o kolejnych przepływach (nieparzyste numery kolumn: 5, 7, ...) i o czasach ich trwania (parzyste numery kolumn: 4, 6, ...).

### Dostosuj parametry w pliku config.py
Znaczenie parametrów:
#### Parametry dotyczące generowania danych
* processes - ilość procesów generujących danych(nie powinna przekraczać liczby wątków procesora)
* continuePrevious - określa czy zostawić(continuePrevious=True) poprzednio wygenerowane dane, czy usunąć
* NsamplesPrior - ilość próbek(ciągów macierzy OD) danych apriorycznych
* NsamplesHist - ilość próbek(ciągów macierzy OD) danych historycznych
* NsamplesReal - ilość próbek(ciągów macierzy OD) danych w czasie rzeczywistym
* dT - przedział czasu w minutach
* N - czas trwania symulacji jako wielokrotność dT
* delta - ilość poprzednich przedziałów czasowych uwzględnionych w dynamicznej macierzy przepływów

* tazPath - ścieżka do pliku zawierającego strefy TAZ
* edgesPath - ścieżka do pliku zawierającego krawędzie na których będzie się odbywał pomiar natężeń ruchu

* ODgenPatternPath - ścieżka do pliku do którego zapisywana będzie bazowa macierz przepływu(zdefiniowana w pliku OD.txt) w formacie .pkl
* MINprior - dolna granica zakresu z którego losowane są macierze OD ze zbioru danych apriorycznych
* MAXprior - górna granica zakresu z którego losowane są macierze OD ze zbioru danych apriorycznych
* MINreal - dolna granica zakresu z którego losowane są macierze OD ze zbioru danych historycznych i czasu rzeczywistego
* MAXreal - górna granica zakresu z którego losowane są macierze OD ze zbioru danych historycznych i czasu rzeczywistego

* rootDataDir - ścieżka do katalogu, w którym będą umieszczone wygenerowane dane

* InputDirPrior - katalog, w którym będą zapisywane macierze przepływu należące do zbioru danych apriorycznych
* SumoDirPrior - katalog, w którym będą zapisywane pliki konieczne do uruchomienia symulacji generującej natężenia ruchu należące do zbioru danych apriorycznych
* OutputDirPrior - katalog, w którym będą zapisywane natężenia ruchu wygenerowane przez symulator SUMO(dla zbioru danych apriorycznych)

* InputDirHist - katalog, w którym będą zapisywane macierze przepływu należące do zbioru danych historycznych
* SumoDirHist - katalog, w którym będą zapisywane pliki konieczne do uruchomienia symulacji generującej natężenia ruchu należące do zbioru danych historycznych
* OutputDirHist - katalog, w którym będą zapisywane natężenia ruchu wygenerowane przez symulator SUMO(dla zbioru danych historycznych)

* InputDirRealTime - katalog, w którym będą zapisywane macierze przepływu należące do zbioru danych w czasie rzeczywistym
* SumoDirRealTime - katalog, w którym będą zapisywane pliki konieczne do uruchomienia symulacji generującej natężenia ruchu należące do zbioru danych w czasie rzeczywistym
* OutputDirRealTime - katalog, w którym będą zapisywane natężenia ruchu wygenerowane przez symulator SUMO (dla zbioru danych w czasie rzeczywistym)

* sumoDir - katalog zawierający pliki .taz.xml, .net.xml oraz z katalog /detectors
* netFile - ścieżka do pliku .net.xml

* od2tripsOptions - opcje zastosowane w programie od2trips
* durouterOptions - opcje zastosowane w programie duarouter
* sumoOptions - opcje zastosowane w programie sumo

#### Parametry dotyczące trenowania sieci CNN
* scaling - określa czy dane są skalowane
* epochs - ilość epok, którą trwa trening
* early_stopping - ilość epok po których powinna nastąpić poprawa funkcji strat (jeśli f. strat się nie poprawi trening jest przerywany)
* batch_size - ilość próbek po których aplikowana jest poprawka wag całej sieci neuronowej
* numCNNFilters - ilość filtrów w poszczególnych warstwach konwolucyjnych
* ksi, eta - parametry warstwy SAAF zaimplementowanej w pliku saaf.py

#### Parametry dotyczące algorytmu genetycznego Offline
* Noffline - ilość osobników w populacji
* NGENoffline - ilość generacji
* crossprob_offline - prawdopodobieństwo krzyżowania
* mutprob_offline - prawdopodobieństwo mutacji

#### Parametry dotyczące algorytmu genetycznego Online
* Nonline - ilość osobników w populacji
* NGENonline - ilość generacji
* crossprob_online - prawdopodobieństwo krzyżowania
* mutprob_online - prawdopodobieństwo mutacji
* t - numer przedziału czasowego dla którego estymowana jest macierz przepływu

## Uruchomienie oprogramowania
0. Przetestuj konfigurację oprogramowania za pomocą polecenia `python simTestOD.py` (aby posłużyć się danymi z pliku **OD.txt**) lub `python simTestOD.py` (aby posłużyć się danymi z plików **DOD.txt** i **ODpairs.txt**). Jeśli w symulacji dostrzeżono pewne nieprawidłowości, to należ poprawić pewne elementy konfiguracji np. sieć **map.net.xml** za pomocą programu NETEDIT lub plik config.py.
1. Wygeneruj dane za pomocą polecenia: `python generateDataFromOD.py` (aby posłużyć się danymi z pliku OD.txt) lub `python generateDataFromDOD.py` (aby posłużyć się danymi z plików DOD.txt i ODpairs.txt).
2. Wytrenuj i przetestuj model CNN za pomocą polecenia: `python trainCNN.py`
3. Uruchom algorytm genetyczne poleceniem: `python GA.py`, aby uzyskać wzorcową dynamiczną macierz przepływu
4. Uruchom algorytm genetyczny poleceniem: `python GAonline.py`, aby uzyskać macierz przepływu dla danego przedziału czasowego

## Funkcjonalności
* Generowanie danych
* Konfigurowanie, trenowanie i testowanie modeli sieci neuronowych
* Poszukiwanie dynamicznej macierzy przepływu

## TODO
* Graficzny interfejs użytkownika
* Refaktoryzacja kodu
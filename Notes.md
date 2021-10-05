# To be done
- Include an argument which allows to choose between HM flux and DB flux in NEOS.
- Include the chi2 computation, using the covariance matrix.
- Put in common how DB and NEOS adapt to a custom binning.

Miscellanious:
- Edit "Python implementation/EventExpectationPlots.py" so that it looks as well as GlobalFit plots.

### Notes

He descobert que el Poisson que faig dóna molt més pes al EH1 que als altres, i en concret, al EH3 li'n dona molt poca. Això fa que, al final, el fit es faci al que fita millor a l'EH1, i que l'EH3 no quedi ben fitat.

Crec que el més complicat de recrear hauria de ser la matriu de reconstrucció/resposta. Tenen fets uns slices en el https://zenodo.org/record/1286994#.YNnMRhMza-o on pots comparar. Bàsicament són dues gaussianes.
També s'haurà d'integrar per L diferents (tenint en compte que L és proper, veiem el detector amb un cert gruix), però per fer-ne una primera anàlisi no caldria. S'hauria de mirar si es pot fer de manera analítica la integral!
A banda d'aquests punts, crec que tots és, si fa no fa, força semblant a DayaBay.

---

02 juliol

He implementat la integració numèrica dela probabilitat entre L^2 (no existeix analítica).
Caldria comprovar que està bé! En concret, tinc dubtes amb si s'ha de dividir entre W.
Tinc les notes al despatx on hi ha feta la integració "analítica". Revisar bé.
Una manera de comprovar-ho seria fer un binning molt fi, i mirar si convergeix (sembla que sí).
Una altra manera de comprovar-ho és fer el detector "puntual", és a dir, prendre W << L.

Per altra banda, la funció de Etrue to Erec la tinc al portàtil.
He de fer una taula amb binning de 50keV aplicant-hi aquesta funció, i escriure-la a NEOS/Data/ReconstructMatrix.dat. S'ha de mirar quin format té la mateixa taula a DayaBay.

---

06 de juliol
He de fer una interpolació del histograma que hi ha a NEOSParameters.py.
Un cop fet això, el que hem de fer és implementar aquesta funció del flux dins de la funció que calcula el nombre d'esdeveniments esperats.
Això ho hauríem de fer tant a DayaBay com a NEOS (?). Aleshores, un cop impementat això, hem de fer el càlcul del nombre esperat de neutrins per DayaBay i per NEOS, per 3neutrins i per 4, i calcular el ràtio que fa Kopp.

---

07 de juliol
El que s'hauria de fer primer de tot és convertir les dades (esdeveniments per dia per 100 keV) en dades absolutes,
és a dir, multiplicar-les totes per 100 keV (i s'hauria de mirar el tema dels dies).
Després s'han de normalitzar la nostra predicció per què quadri amb la predicció.
Finalment, s'ha de comparar la nostra predicció amb la seva. Probablement s'haurà de retallar la matriu de reconstrucció.

---

01 de setembre
Ja he aterrat i sembla que en general tot és força correcte. Crec que el problema principal recau en la manera com hem reconstruït la matriu de reconstrucció. Ara per ara, sembla que té una cua massa curta a eprompt baixes i que el centre de la gaussiana petita no acaba de quadrar.
El que hauria de fer és penjar el que he fet al portàtil (el jupyter notebook d'ajustar i escriure la matriu) al dropbox i implementar-ho al programa principal.

---

03 de setembre
En Jordi diu que no ho veu gaire malament, quedem a l'espera de comentar-ho amb en Carlos (sobretot hi ha el darrer bin, que queda una mica raro perquè la predicció està per sota del background).
Què queda per fer ara? El global fit dels 4 experimental halls alhora.
Quins problemes hi ha?
  - El nombre de bins de DayaBay no concorda amb el nombre de bins de NEOS. És necessari que coincideixin per poder utilitzar els mateixos nuissance parameters per tots ells. Per solucionar-ho, cal retallar bins de DayaBay (començar a 1.3 i acabar a 7.0 o 6.9), retallar els de NEOS (començar a 1.3 i acabar a 6.9), i juntar els de NEOS de manera que vagin cada 0.2 i no cada 0.1.
  - El nombre de dades de NEOS és molt menor que el nombre de dades de DayaBay. Això fa que, segons el tractament que vam fer per DB, NEOS no influeixi gairebé res en el càlculs dels nuissance parameters (i aleshores no tindrà gaire importància?).
Què queda per fer?
  - A banda de resoldre aquests dos problemes, cal implementar la darrera funció, la que calculi el chi2.
  - Fer diferents càlculs de chi2 amb diferents valors dels paràmetres.

---

06 de setembre
Si estic fent un global fit, la cosa seria utilitzar el HubberMullerFlux tant per DayaBay com per NEOS, no?

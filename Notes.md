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

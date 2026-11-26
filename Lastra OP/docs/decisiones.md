# Decisiones de diseño

## Resultado común
Esquina noroeste, costo mínimo, Vogel y programación lineal devuelven un `TransportSolution`
(`core/models.py`) con la asignación, el costo total, los pasos (`steps`) y las celdas básicas.
La interfaz muestra `steps` sin recalcular nada.

## Balanceo
Los métodos exigen un problema balanceado y lanzan `ValidationError` si no lo está.
`balancear(problema, costo_ficticio)` agrega:
- un **destino ficticio** cuando la oferta supera la demanda;
- un **origen ficticio** cuando la demanda supera la oferta (demanda insatisfecha).

`costo_ficticio` es un parámetro (número o vector) con valor por defecto 0, para poder
penalizar la demanda insatisfecha si la empresa lo decide.

## Reglas de desempate
- **Costo mínimo:** entre celdas activas de igual costo, la primera en orden de fila
  (fila menor y, dentro de ella, columna menor).
- **Vogel:** penalización = diferencia entre los dos costos activos más bajos de la línea
  (si queda una sola celda, ese costo). Gana la mayor penalización; si una fila y una
  columna empatan, gana la fila; entre líneas empatadas, la de menor índice. Dentro de la
  línea, la celda de menor costo y, si empatan, la de menor índice.

## Degeneración
Si un paso agota a la vez la oferta y la demanda, se tacha solo la fila (o la columna si
ya no quedan filas). Así toda solución inicial tiene m + n − 1 celdas básicas, con algunas
asignaciones en 0 si hace falta.

## Resultados del caso base (miles de COP)
| Método | Costo | Sobrecosto vs. óptimo |
|---|---|---|
| Esquina noroeste | 8.100 | +2.250 (38,5 %) |
| Costo mínimo | 6.350 | +500 (8,5 %) |
| Vogel | 5.850 | 0 |
| Programación lineal | 5.850 | — |

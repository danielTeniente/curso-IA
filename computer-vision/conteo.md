# Conteo de objetos
El conteo de objetos en un video requiere de tratamiento especial. No basta con detectar las formas en una imagen, hay que dar seguimiento a cada forma a través del video, de tal forma que no contemos el mismo objeto dos veces.

## Pasos
1. Detectar las formas en cada frame del video.
2. Calcular la distancia entre cada forma detectada en el frame actual y las formas detectadas en el frame anterior.
3. Si la distancia entre dos formas es menor a un umbral, se consideran la misma forma.
4. Si la distancia entre dos formas es mayor al umbral, se consideran formas diferentes.
5. Los objetos deben atravesar una zona de salida para ser contados. Si una forma entra en la zona de salida, se considera que el objeto ha salido del video, su registro puede ser eliminado.

La zona de salida tiene importancia a la hora de mapear flujos urbanos, porque también permite determinar la dirección de los objetos. Por ejemplo, si las personas o vehículos se mueven de norte a sur.

## Notebooks
1. [Detectar el mismo en frames consecutivos](./conteo/paso1/conteo1.md)
2. ?
* [Establecer la zona de salida](./conteo2.ipynb)
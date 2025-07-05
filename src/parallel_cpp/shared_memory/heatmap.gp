# heatmap.gnuplot

set terminal pngcairo size 800,800 enhanced font 'Arial,10'
set output 'heatmap.png'


unset key
set size ratio 0  # Cuadrado
set palette defined ( 0 "web-green", 1 "dark-blue", 2 "red" ) 
set cblabel 'Potencial eléctrico (V)'
set colorbox

# Para que Gnuplot interprete la matriz como una imagen
plot 'matriz.txt' matrix with image


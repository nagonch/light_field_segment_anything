mkdir MMSPG

wget http://plenodb.jpeg.org/lf/epfl/urban/bikes_4d_lf_mat.zip -O MMSPG/bikes.zip
unzip MMSPG/bikes.zip
mv Urban/Bikes.mat MMSPG/bikes.mat
rm MMSPG/bikes.zip
rm -r Urban

wget http://plenodb.jpeg.org/lf/epfl/mirrors_and_transparency/fountain_and_bench_4d_lf_mat.zip -O 'MMSPG/fountain_&_bench.zip'
unzip 'MMSPG/fountain_&_bench.zip'
mv 'Mirrors_and_Transparency/Fountain_&_Bench.mat' 'MMSPG/fountain_&_bench.mat'
rm 'MMSPG/fountain_&_bench.zip'
rm -r Mirrors_and_Transparency

wget http://plenodb.jpeg.org/lf/epfl/people/friends_1_4d_lf_mat.zip -O MMSPG/friends_1.zip
unzip MMSPG/friends_1.zip
mv 'People/Friends_1.mat' 'MMSPG/friends_1.mat'
rm MMSPG/friends_1.zip
rm -r People

wget http://plenodb.jpeg.org/lf/epfl/people/sophie_and_vincent_3_4d_lf_mat.zip -O 'MMSPG/sophie_&_vincent_3.zip'
unzip 'MMSPG/sophie_&_vincent_3.zip'
mv 'People/Sophie_&_Vincent_3.mat' 'MMSPG/sophie_&_vincent_3.mat'
rm 'MMSPG/sophie_&_vincent_3.zip'
rm -r People

wget http://plenodb.jpeg.org/lf/epfl/people/sphynx_4d_lf_mat.zip -O MMSPG/sphynx.zip
unzip MMSPG/sphynx.zip
mv 'People/Sphynx.mat' 'MMSPG/sphynx.mat'
rm MMSPG/sphynx.zip
rm -r People

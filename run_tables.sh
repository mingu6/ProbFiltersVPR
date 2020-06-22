ParticleFilter.py  
tables.py -f tables1.txt 
ParticleFilter.py  
tables.py -f tables2.txt 
ParticleFilter.py  
tables.py -f tables3.txt 
ParticleFilter.py  
tables.py -f tables4.txt 
ParticleFilter.py  
tables.py -f tables5.txt 
echo "Generate PR curve..." 
PR_curves.py -t 3 -R 5 
PR_curves.py -t 5 -R 10 
echo "All experiments complete! All results generated and saved successfully." 

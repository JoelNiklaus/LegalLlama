# Explanation
We used a Ubuntu VM with a RTX A6000 and 100 GB RAM to run the code. 
First Generate the data with GenData.py.
Then one can run the e5 scripts.

## BM25 (elasticsearch)
To run the BM25 scripts, one needs to start an elasticsearch instance first. 
To do so use the following commands in your commandshell directly:<br>

$ wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.9.2-linux-x86_64.tar.gz<br>
$ wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.9.2-linux-x86_64.tar.gz.sha512<br>
$ tar -xzf elasticsearch-oss-7.9.2-linux-x86_64.tar.gz<br>
$ sudo chown -R daemon:daemon elasticsearch-7.9.2/<br>
$ shasum -a 512 -c elasticsearch-oss-7.9.2-linux-x86_64.tar.gz.sha512<br>

Now to ***avoid errors***, we need to increase the default max_clause_count of elasticsearch.
For this add the following line to the elasticsearch.yml file in the elasticsearch config folder you just installed: <br>
indices.query.bool.max_clause_count: 100000

Now start the deamon process in the background:<br>
$ sudo -H -u daemon elasticsearch-7.9.2/bin/elasticsearch &

Give it some time to let it start and then check whether it's up and running with:<br> 
$ ps -ef | grep elasticsearch<br>
and <br>
$ curl -sX GET "localhost:9200/"

From there you can run the BM25 python scripts.

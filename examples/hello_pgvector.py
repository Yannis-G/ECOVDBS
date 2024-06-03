import psycopg
from pgvector.psycopg import register_vector

with psycopg.connect("host=localhost port=5432 dbname=postgres user=postgres password=pwd") as conn:
    conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(conn)
    conn.execute('DROP TABLE IF EXISTS items')
    conn.execute('CREATE TABLE IF NOT EXISTS items (id bigserial PRIMARY KEY, embedding vector(3))')
    conn.execute("INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');")
    res = conn.execute("SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;").fetchall()
    print(res)
    res = conn.execute("SELECT embedding <-> '[3,1,2]' AS distance FROM items;").fetchall()
    print(res)
    conn.execute('CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)')
    res = conn.execute("SELECT embedding <-> '[3,1,2]' AS distance FROM items;").fetchall()
    print(res)

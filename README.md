# FREL
Financial Reinforcement with Endogenous Learning

# Structure
FREL/
├── config/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── synth.py
│   ├── reward.py
│   └── model.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_synth.py
│   ├── test_reward.py
│   └── test_model.py
└── dev/
    └── main.py

# Examples
model = build_<name>(price_shape=(200,5),
                     meta_len=10,
                     d_model=128,
                     nhead=4,
                     blocks=4,
                     dropout=0.1)

for chunk in parquet_chunks:
    ds = tf.data.Dataset.from_tensor_slices(chunk).batch(1024)
    model.fit(ds, epochs=1)

# Other Model useage
model = build_patchtst()
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(1e-3), jit_compile=True)

# stream 12 M rows at a time
for chunk in pd.read_parquet("samples_320M.parquet", chunksize=12_000_000):
    ds = tf.data.Dataset.from_tensor_slices(
        ({'price': np.stack(chunk['close_scaled']),   # already (N,200,5)
          'meta': chunk[meta_cols].astype('float32')},
         chunk['y'].astype('float32'))
    ).batch(1024)
    model.fit(ds, epochs=1)

model = build_nhits()
model.compile(loss='mse', optimizer='adam', jit_compile=True)

for df_chunk in pd.read_parquet('samples_320M.parquet', chunksize=12_000_000):
    ds = tf.data.Dataset.from_tensor_slices(
        ({'price': np.stack(df_chunk['close_scaled']),
          'meta':  df_chunk[meta_cols].astype('float32')},
         df_chunk['y'].astype('float32'))
    ).batch(1024)
    model.fit(ds, epochs=1)
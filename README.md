# SOX GPU

Generating SOX-style spectrograms on a gpu

- sox: https://github.com/chirlu/sox
- pysox: https://github.com/marl/pysox
- audio samples: 
	* https://storage.googleapis.com/dse-soundhub-public/data/sample_audio/20230522_000000.flac
	* https://storage.googleapis.com/dse-soundhub-public/data/sample_audio/20230526_000000.flac
- pnw-cnet-model: https://storage.googleapis.com/dse-soundhub-public/models/pnw-cnet/PNW-Cnet_v4_TF.h5


### PNW MODEL

** For PNW we need**

```bash
export TF_USE_LEGACY_KERAS=1
```

This environment variable forces TensorFlow 2.16+ to use the legacy Keras implementation instead of the new Keras 3, which maintains compatibility with H5 models saved in older TensorFlow versions. The newer Keras 3 has stricter input shape validation that can break when loading older model files.

---

## QUICK START

Usage example

---

## DOCUMENTATION

API Docs

---

## STYLE-GUIDE

Following PEP8. See [setup.cfg](./setup.cfg) for exceptions. Keeping honest with `pycodestyle .`

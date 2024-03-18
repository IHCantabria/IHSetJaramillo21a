
# IHSetJaramillo21a
Python package to run and calibrate Jaramillo et al. (2021a) equilibrium-based shoreline evolution model.

## :house: Local installation
* Using pip:
```bash

pip install git+https://github.com/defreitasL/IHSetJaramillo21a.git

```

---
## :zap: Main methods

* [jaramillo21a](./IHSetJaramillo20/jaramillo21a.py):
```python
# model's it self
jaramillo21a(E, dt, a, b, cacr, cero, Yini, vlt)
```
* [cal_Jaramillo21a](./IHSetJaramillo20/calibration.py):
```python
# class that prepare the simulation framework
cal_Jaramillo21a(path)
```



## :package: Package structures
````

IHSetJaramillo21a
|
├── LICENSE
├── README.md
├── build
├── dist
├── IHSetJaramillo21a
│   ├── calibration.py
│   └── jaramillo21a.py
└── .gitignore

````

---

## :incoming_envelope: Contact us
:snake: For code-development issues contact :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria)

## :copyright: Credits
Developed by :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria).

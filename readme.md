## ⚽ Player Similarity Finder with AI

> **Sistema de recomendación inteligente de jugadores de fútbol usando Machine Learning y clustering avanzado**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Resumen del Proyecto

**Player Similarity Finder** es una aplicación web interactiva que utiliza técnicas avanzadas de Machine Learning para encontrar jugadores de fútbol similares basándose en sus estadísticas y rendimiento. El sistema analiza datos de las **Big 5 ligas europeas** (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) de la temporada 2024/2025.

### 🎯 Características Principales

- **🔍 Búsqueda inteligente**: Encuentra jugadores similares usando algoritmos KNN
- **📊 Clustering avanzado**: Agrupa jugadores por perfiles estadísticos (K-Means)
- **📈 Visualizaciones profesionales**: Radar charts, heatmaps y gráficos comparativos
- **🎯 Filtros avanzados**: Por liga, nacionalidad, edad, posición y minutos jugados
- **📱 Interfaz moderna**: App web responsive con Streamlit
- **💾 Exportación de datos**: Descarga resultados en CSV para análisis posterior

---

## 🚀 Demo en Vivo

### Capturas de Pantalla

| Vista Principal | Búsqueda de Similares | Radar Chart |
|----------------|----------------------|-------------|
| ![Dashboard](images/dashboard.png) | ![Search](images/search.png) | ![Radar](images/radar.png) |

### Casos de Uso Reales

```bash
# Ejemplos que puedes probar en la app:
🔎 "Erling Haaland" → Encuentra goleadores similares
🔎 "Luka Modric" → Descubre creativos parecidos  
🔎 "Kylian Mbappe" → Jugadores con perfil similar
🔎 "Jude Bellingham" → Box-to-box prometedores
```

---

## 📁 Estructura del Proyecto

```
player-similarity-finder/
├── 📊 data/
│   ├── raw/                          # Datos originales de FBref
│   └── processed/                    # Datos limpios con clusters
├── 📓 notebooks/
│   ├── 01_EDA.ipynb                 # Análisis exploratorio
│   ├── 02_Clustering.ipynb          # K-Means y DBSCAN
│   └── 03_Player_Similarity.ipynb   # Sistema de recomendación
├── 💻 src/
│   └── streamlit_app.py             # Aplicación web principal
├── 📋 reports/
│   └── figures/                     # Gráficos exportados
├── 📄 requirements.txt              # Dependencias Python
├── 📖 README.md                     # Este archivo
└── 📝 Glosario.md                   # Explicación de métricas
```

---

## 🛠️ Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (opcional)

### Instalación Rápida

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/player-similarity-finder.git
   cd player-similarity-finder
   ```

2. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecuta la aplicación:**
   ```bash
   streamlit run src/streamlit_app.py
   ```

4. **Abre tu navegador en:**
   ```
   http://localhost:8501
   ```

### Dependencias Principales

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 📊 Metodología y Algoritmos

### 1. **Análisis Exploratorio de Datos (EDA)**
- Limpieza y validación de más de 2700 registros
- Análisis de distribuciones y correlaciones
- Detección de outliers y jugadores únicos
- Normalización de métricas por 90 minutos

### 2. **Feature Engineering**
- Selección de 12 métricas clave:
    - **Ofensivas**: Goles/90, Asistencias/90, G+A/90
    - **Expected metrics**: xG/90, xAG/90, npxG/90
    - **Progresión**: PrgC, PrgP, PrgR
    - **Contextuales**: Edad, Tarjetas

### 3. **Clustering (K-Means)**
- Optimización del número de clusters (K=6)
- Validación con Silhouette Score y método del codo
- Identificación de 6 perfiles de jugadores:
    - Goleadores elite
    - Creativos/Asistidores
    - Box-to-box completos
    - Defensas progresivos
    - Mediocampistas defensivos
    - Jugadores de bajo impacto

### 4. **Sistema de Recomendación (KNN)**
- Algoritmo K-Nearest Neighbors con distancia euclidiana
- Búsqueda flexible por nombre parcial
- Puntuación de similitud normalizada (0-100%)

---

## 🎮 Guía de Uso

### Funcionalidades Principales

#### 🔍 **Búsqueda de Jugadores Similares**
1. Escribe el nombre del jugador en la barra lateral
2. Ajusta el número de similares a mostrar (3-15)
3. Aplica filtros opcionales (liga, edad, nacionalidad)
4. Explora los resultados con visualizaciones

#### 📊 **Filtros Avanzados**
- **Por Liga**: Premier League, La Liga, Serie A, etc.
- **Por Nacionalidad**: Más de 80 países representados
- **Por Edad**: Rango personalizable + filtro U23
- **Por Minutos**: Filtrar jugadores con participación mínima

#### 📈 **Rankings Dinámicos**
- Ordena por cualquier métrica estadística
- Top goleadores, asistidores, progresión, etc.
- Exporta rankings personalizados en CSV

#### 🎨 **Visualizaciones**
- **Radar Charts**: Comparación visual de perfiles
- **Gráficos de barras**: Métricas lado a lado
- **Tablas interactivas**: Información detallada

---

## 📈 Casos de Uso Profesionales

### 🎯 **Para Clubs y Ojeadores**
- Identificar alternativas económicas a fichajes objetivo
- Descubrir talentos emergentes con perfiles similares a estrellas
- Análisis de mercado y scouting automatizado

### 📚 **Para Analistas y Académicos**
- Investigación en ciencia deportiva
- Validación de modelos predictivos
- Análisis comparativo de ligas y equipos

### 🎮 **Para Aficionados**
- Descubrir jugadores parecidos a sus favoritos
- Entender estilos de juego mediante datos
- Debates informados con estadísticas avanzadas

---

## 🔬 Resultados y Validación

### Métricas de Calidad del Modelo

| Algoritmo | Métrica | Resultado |
|-----------|---------|-----------|
| K-Means   | Silhouette Score | 0.232 |
| K-Means   | Inercia | Optimizada |
| KNN       | Vecinos | 11 (configurado) |
| PCA       | Varianza Explicada | 59.2% (2 componentes) |

### Ejemplos de Validación

- **Haaland → Similares**: Lewandowski, Osimhen, Isak
- **Modrić → Similares**: De Bruyne, Verratti, Pedri
- **Mbappé → Similares**: Vinicius Jr., Rashford, Leão

---

## 📊 Datos y Fuentes

### Dataset
- **Fuente**: [FBref.com](https://fbref.com)
- **Temporada**: 2024/2025
- **Ligas**: Big 5 Europeas
- **Jugadores**: +2700 registros únicos
- **Métricas**: +30 estadísticas por jugador

### Métricas Principales
- **Per90**: Valores normalizados por 90 minutos
- **Expected Stats**: xG, xAG (estadísticas esperadas)
- **Progressive**: Pases y jugadas progresivas
- **Defensive**: Intercepciones, entradas, duelos

---

## 🤝 Contribuir al Proyecto

### ¿Cómo Contribuir?

1. **Fork** el repositorio
2. **Crea** una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Abre** un Pull Request

### Ideas para Contribuciones

- 🌍 **Internacionalización**: Traducciones a otros idiomas
- 📱 **Mobile UX**: Mejoras para dispositivos móviles
- 🔄 **Datos en tiempo real**: Integración con APIs live
- 🤖 **ML avanzado**: Modelos predictivos de rendimiento
- 🎨 **Visualizaciones**: Nuevos tipos de gráficos

---

## 📞 Contacto y Soporte

### Autor
- **Nombre**: [Tu Nombre]
- **LinkedIn**: [linkedin.com/in/tu-perfil](https://linkedin.com/in/tu-perfil)
- **Email**: tuemail@ejemplo.com
- **GitHub**: [github.com/tu-usuario](https://github.com/tu-usuario)

### Soporte
- 🐛 **Issues**: [GitHub Issues](https://github.com/tu-usuario/player-similarity-finder/issues)
- 💬 **Discusiones**: [GitHub Discussions](https://github.com/tu-usuario/player-similarity-finder/discussions)
- 📧 **Email**: Contacto directo para colaboraciones

---

## 📄 Licencia

Este proyecto está licenciado bajo la **MIT License**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025 Tu Nombre

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Agradecimientos

- **FBref.com**: Por proporcionar datos estadísticos detallados
- **Streamlit**: Por la plataforma de desarrollo web
- **scikit-learn**: Por los algoritmos de Machine Learning
- **Comunidad Open Source**: Por las librerías y herramientas utilizadas

---

## 📈 Roadmap Futuro

### Versión 4.0 (Q1 2025)
- [ ] Integración con APIs en tiempo real
- [ ] Modelos predictivos de rendimiento
- [ ] Sistema de alertas personalizadas
- [ ] Dashboard para equipos profesionales

### Versión 5.0 (Q2 2025)
- [ ] Análisis de video automático
- [ ] Métricas tácticas avanzadas
- [ ] Inteligencia artificial generativa
- [ ] Plataforma SaaS para clubs

---

<div align="center">

**⭐ Si este proyecto te ha sido útil, considera darle una estrella en GitHub ⭐**

[⬆️ Volver arriba](#-player-similarity-finder-with-ai)

</div>

---

*Última actualización: Octubre 2025*

# Business-Intelligence
Repositorio dedicado a la asignatura Inteligencia de Negocio del Grado en Ingeniería Informática (UGR). 2018-2019

- **Práctica 1:** Análisis Predictivo Empresarial Mediante Clasificación
  - Se considerarán al menos seis algoritmos de clasificación distintos. Se valorará la selección
justificada de estos algoritmos en función de las caracterı́sticas del conjunto de datos ası́
como la elección de variedad de tipos de representación (árboles, reglas, redes neuronales,
etc.).  
  - Toda la experimentación se realizará con validación cruzada de 5 particiones. Para sus-
tentar el análisis comparativo se emplearán tablas de errores, matrices de confusión y
curvas ROC. Además de la precisión, se añadirán otras medidas de rendimiento como
TPR, TNR, AUC, Valor-F 1 , G-mean y de complejidad del modelo (número de hojas,
número de reglas, número de nodos, etc.).  
  - Todos los análisis de resultados serán comparativos, de forma que se estudien los pros y
contras de cada representación y/o de cada algoritmo. La documentación deberá incluir
al menos una tabla resumen que incluya los resultados medios de todos los algoritmos
analizados. El análisis no podrá reducirse a una simple lectura de los resultados obtenidos.
El alumno deberá formular y argumentar hipótesis sobre las razones de cada resultado. En
este problema, ¿por qué el algoritmo X funciona mejor que el Y? ¿Por qué la representación
X presenta ciertas ventajas respecto a la Y?.  
  - Se probarán configuraciones alternativas de los parámetros de los algoritmos empleados
justificando los resultados obtenidos. Por ejemplo, ¿puedo evitar o paliar el sobreapren-
dizaje ajustando los parámetros? ¿Puedo obtener modelos más fácilmente interpretables
sin sacrificar excesiva precisión? Para realizar este análisis, se incluirán tablas comparati-
vas con los resultados del algoritmo con parámetros o configuración por defecto y con las
distintas variaciones estudiadas. Si el análisis es suficientemente completo, no es necesario
estudiar todos los algoritmos analizados, se pueden escoger solo algunos de ellos.  
  - Se deberán analizar los datos con diferentes gráficas para comprender su naturaleza e
influencia en el proceso de clasificación.
  - A la luz de este análisis, se deberá estudiar un procesado básico de los datos que mejore
la predicción (por ejemplo, eliminar alguna caracterı́stica por razón justificada, agrupar
los valores posibles de una caracterı́stica, eliminar ciertas instancias del conjunto de en-
trenamiento que se consideren erróneas, convertir una caracterı́stica categórica en varias
binarias, imputar valores perdidos, equilibrar el balanceo de clases...). Deberán justifi-
carse las acciones tomadas y analizar porqué determinado procesado funciona mejor en
un determinado tipo de algoritmo. Si no se consigue mejorar la predicción, se podrá al
menos describir los procesados que se han probado y los resultados obtenidos. De nuevo,
se requiere una tabla resumen que muestre los resultados antes y después de los diferentes
procesados de datos.
  - Basado en todo lo anterior, se deberán extraer conclusiones sobre los factores que deter-
minan que una noticia sea popular. Para llegar a estas conclusiones, se pueden analizar
los modelos legibles generados (por ejemplo, árboles de decisión, conjuntos de reglas o
regresiones lineales), analizar la importancia de cada caracterı́stica en el proceso de clasi-
ficación y visualizar los resultados de predicción de los modelos sobre diferentes casos de
entrada (What-If Analysis).

# CardioRed · interfaz web (Angular)

Interfaz gráfica del proyecto de predicción de enfermedad cardíaca con redes neuronales. Sustituye a los
gráficos de matplotlib del script original por una aplicación Angular con tres páginas:

| Ruta          | Página          | Contenido                                                                                                                         |
| ------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `/`           | Inicio / panel  | KPIs de test (count-up) con media ± desv. de CV, comparación justa con regresión logística (umbral 0,5), curvas de pérdida con la mejor época, matriz de confusión, curva ROC y distribución de clases |
| `/prediccion` | Predicción      | Formulario en 3 pasos generado desde `/api/features`, medidor animado 0-100 % de probabilidad de enfermedad, clase predicha (umbral `metrics.threshold`) y banda de riesgo (cortes fijos 33/66 %), avisos médicos |
| `/modelo`     | Modelo y datos  | Arquitectura del MLP, nota sobre la etiqueta objetivo, diccionario de variables con estadísticas y metodología                     |

> **Aviso:** herramienta educativa. Los resultados no sustituyen el diagnóstico de un profesional sanitario.
> El modelo predice la probabilidad de enfermedad cardíaca (clase 1). La columna `output` del CSV de Kaggle
> está invertida respecto a UCI, por lo que el modelo se entrena con `enfermedad = 1 − output`.

## Requisitos

- Node.js `^20.19`, `^22.12` o `>=24` (probado con 22.16) y npm 11.
- La API FastAPI del repositorio escuchando en `http://localhost:8000` (o usar el modo de datos de ejemplo).

## Puesta en marcha

```bash
# 1) API (desde la raíz del repositorio, con su entorno virtual activado)
python -m uvicorn api.main:app --port 8000

# 2) Interfaz
cd web
npm ci
npm start            # http://localhost:4200
```

`npm start` usa `proxy.conf.json`, que redirige `/api/*` a `http://localhost:8000`, así que no hace falta
configurar CORS en desarrollo. Si la API no responde, la interfaz muestra un aviso con botón **Reintentar**.

### Sin backend (datos de ejemplo)

```bash
npm run start:mocks
```

Activa `useMocks` (`src/environments/environment.mocks.ts`): un interceptor HTTP responde con los fixtures de
`src/app/mocks/fixtures.ts` (métricas e historial reales del entrenamiento v2.0.0). La predicción en este modo es
una heurística simulada, **no** el modelo real.

## Scripts

| Comando               | Descripción                                           |
| --------------------- | ----------------------------------------------------- |
| `npm start`           | Servidor de desarrollo con proxy a la API             |
| `npm run start:mocks` | Servidor de desarrollo con datos de ejemplo           |
| `npm run build`       | Compilación de producción en `dist/web/browser`       |
| `npm test`            | Tests unitarios (Vitest + jsdom, sin navegador)       |

## Configuración

- `src/environments/environment*.ts`: `apiBaseUrl` (por defecto `/api`) y `useMocks`.
- En producción, sirve `dist/web/browser` detrás del mismo dominio que la API (o un proxy inverso que enrute
  `/api` a FastAPI), o cambia `apiBaseUrl` a la URL absoluta de la API y habilita CORS en ella.

## Stack

- Angular 21.2 (componentes standalone, signals, control de flujo `@if/@for`, rutas lazy, zoneless).
  Se usa Angular 21 y no 22 porque Angular 22 exige Node ≥ 22.22; con un Node más reciente basta `ng update`.
- Tailwind CSS 4.3 vía `@tailwindcss/postcss` (`.postcssrc.json` + `@import 'tailwindcss'` en `src/styles.css`),
  la integración documentada por Angular y Tailwind.
- Gráficos en SVG hechos a mano (sin dependencias), con tooltip accesible por ratón y teclado.
- Tests con el builder `@angular/build:unit-test` (Vitest 4).

## Tema y colores

La paleta es la **seaborn «flare»** que usaba la matriz de confusión del script original
(`#EDB081 → #6C2B6D`), expuesta como tokens de Tailwind `flare-50…950`, más un acento frío `calma-*`
(familia teal de la app hermana HeartGuard, saturado para pasar las comprobaciones de daltonismo).
Los tokens semánticos (`bg`, `surface`, `fg`, `muted`, `primary`, `series-1/2`…) cambian entre modo claro y
oscuro; el contraste de texto cumple WCAG AA. El modo oscuro se activa por clase (`html.dark`) con el botón
de la cabecera y, por defecto, sigue `prefers-color-scheme`.

## Animaciones

- Línea de ECG animada en la cabecera y corazón latiendo en el logotipo.
- Transiciones entre rutas con la View Transitions API (`withViewTransitions`).
- Entrada escalonada de tarjetas, números con count-up, barras que crecen, curvas que se dibujan.
- Pasos del formulario con `animate.enter` (API nativa de Angular que sustituye a `@angular/animations`).
- Aguja del medidor de riesgo con transición elástica y esqueletos de carga con brillo.
- Con `prefers-reduced-motion: reduce` se desactivan todas las animaciones.

## Estructura

```
src/app/
  core/        modelos del contrato, ApiService, estado de la API, tema, utilidades
  layout/      cabecera (ECG, navegación, tema), pie
  pages/       dashboard, prediction, model-info (cargadas de forma diferida)
  shared/      componentes reutilizables (gráficos, medidor, KPIs, avisos, esqueletos)
  mocks/       fixtures e interceptor de datos de ejemplo
```

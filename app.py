import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Configuración general de la página
st.set_page_config(page_title="Dashboard Tiendas - Visión Gerencial", layout="wide")

# --- Funciones Utilitarias ---
def load_data(path: str = "data.csv") -> pd.DataFrame:
    """Carga el dataset y convierte la columna Date a datetime si existe"""
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def filter_data(df: pd.DataFrame, start_date=None, end_date=None, categories=None, category_col=None, branches=None):
    """Filtra el dataframe según los parámetros seleccionados"""
    filtered = df.copy()
    if "Date" in filtered.columns and start_date and end_date:
        filtered = filtered[(filtered["Date"] >= pd.to_datetime(start_date)) & (filtered["Date"] <= pd.to_datetime(end_date))]
    if category_col and categories:
        filtered = filtered[filtered[category_col].isin(categories)]
    if "Branch" in filtered.columns and branches:
        filtered = filtered[filtered["Branch"].isin(branches)]
    return filtered

def prepare_pca(df: pd.DataFrame, n_components=2):
    """Prepara los datos para análisis PCA"""
    numeric = df.select_dtypes(include='number').dropna(axis=1)
    if numeric.shape[1] >= n_components:
        scaled = StandardScaler().fit_transform(numeric)
        return scaled, numeric
    return None, None

# Carga de datos inicial
df = load_data()
category_col = next((col for col in ["Category", "Product line"] if col in df.columns), None)
categorias = df[category_col].unique() if category_col else []
branches = df["Branch"].unique() if "Branch" in df.columns else []

# Navegación principal
st.sidebar.title("Navegación")
menu = st.sidebar.radio(
    "Seleccione una sección:",
    ["🏠 Inicio", 
     "📌 Variables Clave", 
     "📊 Gráficos Básicos", 
     "📈 Gráficos Compuestos", 
     "🧬 Análisis Multivariado y 3D", 
     "📌 Análisis Complementarios", 
     "📋 Resumen Ejecutivo"]
)

# Sidebar común para todas las secciones
start_date = st.sidebar.date_input("📅 Fecha inicio", df["Date"].min())
end_date = st.sidebar.date_input("📅 Fecha fin", df["Date"].max())
selected_categories = st.sidebar.multiselect("🏷️ Categorías", categorias, default=categorias)
selected_branches = st.sidebar.multiselect("🏬 Sucursales", branches, default=branches)

# Filtrar datos según selecciones
filtered_df = filter_data(df, start_date, end_date, selected_categories, category_col, selected_branches)

# --- SECCIÓN: INICIO ---
if menu == "🏠 Inicio":
    st.title("📊 Dashboard de Ventas - Visión Ejecutiva")
    st.markdown("Bienvenido al centro de análisis estratégico de nuestra cadena de tiendas de conveniencia.")
    st.image("https://miro.medium.com/v2/resize:fit:753/1*1sr0IMJEatpy5v5sZtMyXQ.jpeg", width=800)
    st.markdown("## 🎯 Objetivo General")
    st.markdown("""
    Este dashboard ha sido diseñado para ofrecer a la dirección una **visión clara, rápida y accionable** del comportamiento comercial,
    el rendimiento por tienda y las preferencias del cliente, en base a los datos reales de ventas.
    """)
    st.markdown("## 🗂️ ¿Qué encontrará en este Dashboard?")
    st.markdown("""
    - **📌 Variables Clave:** Revisión inicial y justificación de los datos seleccionados.
    - **📊 Gráficos Básicos:** Exploración simple para comprender la distribución y variabilidad.
    - **📈 Gráficos Compuestos:** Comparaciones y relaciones entre múltiples dimensiones.
    - **🧬 Análisis Multivariado y 3D:** Visualización avanzada y segmentación con técnicas PCA y clustering.
    - **📋 Análisis Complementarios:** Insights sobre calificaciones, pagos y rentabilidad cruzada.
    - **📋 Resumen Ejecutivo:** Conclusiones, indicadores clave y recomendaciones automatizadas.
    """)
    st.markdown("## 🧭 ¿Cómo utilizar este Dashboard?")
    st.markdown("""
    Utilice el **menú lateral** para navegar por las diferentes secciones.  
    En cada vista encontrará:
    - Gráficos interactivos.
    - Explicaciones ejecutivas.
    - Filtros para ajustar los análisis por fecha, tienda o categoría.

    👉 No necesita conocimientos técnicos: cada insight está acompañado de una breve interpretación para facilitar la toma de decisiones.
    """)
    st.markdown("## ✅ Toma de Decisiones Basada en Datos")
    st.markdown("""
    Este dashboard es una herramienta de apoyo a la gestión, diseñada para ayudarle a:
    - **Detectar oportunidades**
    - **Identificar problemas**
    - **Optimizar decisiones comerciales**

    Todo esto a partir de información clara, visual y actualizada.
    """)

# --- SECCIÓN: VARIABLES CLAVE ---
elif menu == "📌 Variables Clave":
    st.title("📌 Selección y Análisis de Variables Clave")
    
    # KPIs principales
    st.markdown("## 📈 Indicadores Clave del Segmento Actual")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧾 Transacciones", len(filtered_df))
    col2.metric("💰 Ventas Totales", f"${filtered_df['Total'].sum():,.2f}")
    col3.metric("🧮 Promedio por Venta", f"${filtered_df['Total'].mean():,.2f}")

    # Tabs para exploración
    tabs = st.tabs(["👁 Vista Previa", "📊 Estadísticas", "📋 Análisis Visual", "🧠 Justificación"])

    # Tab 1 - Vista previa
    with tabs[0]:
        st.subheader("👁 Primeros registros filtrados")
        df_display = filtered_df.copy()
        if "Date" in df_display.columns:
            df_display["Date"] = df_display["Date"].astype(str)
        st.dataframe(df_display.head())

    # Tab 2 - Estadísticas
    with tabs[1]:
        st.subheader("📊 Resumen estadístico de variables numéricas")
        stats = filtered_df.select_dtypes(include='number').describe()
        st.dataframe(stats)

    # Tab 3 - Análisis Visual
    with tabs[2]:
        st.subheader("📋 Visualización rápida de variables clave")

        if "Date" in filtered_df.columns and "Total" in filtered_df.columns:
            st.markdown("#### 📆 Evolución Temporal de Ventas")
            df_time = filtered_df.groupby("Date")["Total"].sum().reset_index()
            fig_time = px.line(df_time, x="Date", y="Total", title="Ventas Totales a lo Largo del Tiempo")
            st.plotly_chart(fig_time, use_container_width=True)

        if "Branch" in filtered_df.columns:
            st.markdown("#### 🏪 Ventas por Sucursal")
            fig_branch = px.bar(filtered_df.groupby("Branch")["Total"].sum().reset_index(), 
                                x="Branch", y="Total", title="Total de Ventas por Sucursal")
            st.plotly_chart(fig_branch, use_container_width=True)

        if category_col:
            st.markdown("#### 🧺 Participación por Categoría")
            fig_cat = px.pie(filtered_df, names=category_col, values="Total", title="Distribución por Categoría")
            st.plotly_chart(fig_cat, use_container_width=True)

    # Tab 4 - Justificación
    with tabs[3]:
        st.subheader("🧠 Justificación de variables clave")
        st.markdown(f"""
        Las siguientes variables han sido seleccionadas como clave en este análisis:

        - **Total**: Representa el monto total por transacción, fundamental para evaluar ingresos.
        - **Quantity**: Cantidad de productos por venta, útil para entender volumen de consumo.
        - **Date**: Permite evaluar patrones temporales y estacionalidad.
        - **Branch**: Compara desempeño entre tiendas.
        - **Gender**: Permite entender el perfil demográfico de clientes.
        - **{category_col or 'otra variable'}**: Da contexto sobre el tipo de producto o servicio adquirido.

        > Estas variables son críticas para entender el rendimiento de ventas, ajustar campañas de marketing, optimizar stock, e incluso rediseñar el layout de tienda o catálogo.
        """)

# --- SECCIÓN: GRÁFICOS BÁSICOS ---
elif menu == "📊 Gráficos Básicos":
    st.title("📊 Exploración Visual Básica")
    
    # Tabs de visualización
    tabs = st.tabs([
        "📈 Histograma de Ventas", 
        "📦 Boxplot por Género", 
        "🔍 Dispersión Total vs Cantidad",
        "⭐ Calificación de Clientes", 
        "👥 Gasto por Tipo de Cliente"
    ])

    # Tab 1 - Histograma de Total
    with tabs[0]:
        st.subheader("📈 Distribución de Ventas Totales")
        st.markdown("Este gráfico permite observar cómo se distribuyen los montos totales de venta por transacción.")
        if "Total" in filtered_df.columns:
            fig = px.histogram(filtered_df, x="Total", nbins=30, title="Distribución de Montos Totales de Venta")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("Se observa una mayor concentración de ventas entre valores bajos, lo que sugiere tickets promedio pequeños en gran parte de las transacciones.")
        else:
            st.warning("No se encontró la columna 'Total' en el dataset.")

    # Tab 2 - Boxplot por Género
    with tabs[1]:
        st.subheader("📦 Comparación de Ventas por Género")
        st.markdown("Este boxplot compara los montos de venta entre hombres y mujeres, permitiendo ver su variabilidad.")
        if "Gender" in filtered_df.columns and "Total" in filtered_df.columns:
            fig2 = px.box(filtered_df, x="Gender", y="Total", title="Distribución de Ventas por Género")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("Ambos géneros presentan una distribución similar en cuanto a ticket de compra, sin diferencias significativas.")
        else:
            st.warning("Faltan columnas 'Gender' o 'Total'.")

    # Tab 3 - Dispersión Quantity vs Total
    with tabs[2]:
        st.subheader("🔍 Relación entre Cantidad y Total Vendido")
        st.markdown("Se visualiza la correlación entre la cantidad de productos y el monto final por transacción.")
        if "Quantity" in filtered_df.columns and "Total" in filtered_df.columns:
            fig3 = px.scatter(
                filtered_df, x="Quantity", y="Total", 
                color="Gender" if "Gender" in filtered_df.columns else None,
                title="Relación entre Cantidad de Productos y Monto Total"
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("Existe una relación directa: a mayor cantidad, mayor total, lo que valida el correcto registro de ventas.")
        else:
            st.warning("Faltan columnas 'Quantity' o 'Total'.")

    # Tab 4 - Distribución de Rating
    with tabs[3]:
        st.subheader("⭐ Distribución de Calificaciones de Clientes")
        st.markdown("El siguiente histograma muestra cómo los clientes valoraron su experiencia.")
        if "Rating" in filtered_df.columns:
            fig4 = px.histogram(filtered_df, x="Rating", nbins=20, title="Distribución de Calificación de Clientes")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("Las calificaciones tienden a concentrarse entre 6 y 9 puntos, indicando un alto grado de satisfacción.")
        else:
            st.warning("No se encontró la columna 'Rating'.")

    # Tab 5 - Gasto por Tipo de Cliente
    with tabs[4]:
        st.subheader("👥 Comparación del Gasto por Tipo de Cliente")
        st.markdown("Compara el comportamiento de gasto entre clientes regulares y miembros.")
        if "Customer type" in filtered_df.columns and "Total" in filtered_df.columns:
            fig5 = px.box(filtered_df, x="Customer type", y="Total", title="Distribución de Gasto por Tipo de Cliente")
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown("Ambos grupos muestran un comportamiento de gasto similar, aunque los miembros parecen tener una ligera mayor dispersión.")
        else:
            st.warning("Faltan columnas 'Customer type' o 'Total'.")

# --- SECCIÓN: GRÁFICOS COMPUESTOS ---
elif menu == "📈 Gráficos Compuestos":
    st.title("📈 Análisis Comparativo de Ventas")
    
    tabs = st.tabs([
        "📊 Tendencia por Categoría",
        "🏬 Ventas por Sucursal",
        "🔥 Sucursal vs Categoría",
        "📉 Costo vs Ganancia Bruta"
    ])

    # Tab 1 - Línea temporal por categoría
    with tabs[0]:
        st.subheader("📊 Evolución de Ventas por Categoría")
        st.markdown("Este gráfico permite observar cómo han variado las ventas (`Total`) para cada categoría a lo largo del tiempo.")

        if category_col and "Date" in filtered_df.columns and "Total" in filtered_df.columns:
            fig_line = px.line(filtered_df, x="Date", y="Total", color=category_col, title="Tendencia de Ventas por Categoría")
            st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("""
            > 📌 **Interpretación**: Este análisis permite detectar estacionalidades o cambios de comportamiento por línea de producto. 
            Si una categoría crece de forma constante, podría ser priorizada en campañas o stock.
            """)
        else:
            st.warning("No se encontraron las columnas necesarias para generar la gráfica.")

    # Tab 2 - Barras por sucursal
    with tabs[1]:
        st.subheader("🏬 Comparación de Ventas por Sucursal")
        st.markdown("Visualización acumulada para entender qué tiendas han generado más ingresos en el período.")

        if "Branch" in filtered_df.columns and "Total" in filtered_df.columns:
            resumen_branch = filtered_df.groupby("Branch")["Total"].sum().reset_index()
            fig_bar = px.bar(resumen_branch, x="Branch", y="Total", title="Ventas Totales por Sucursal")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("""
            > 📌 **Interpretación**: Este gráfico permite identificar cuál sucursal genera mayores ingresos y si hay alguna rezagada. 
            Es útil para enfocar esfuerzos operativos y de marketing.
            """)

    # Tab 3 - Mapa de calor cruzado con gross income
    with tabs[2]:
        st.subheader("🔥 Matriz de Rendimiento por Sucursal y Categoría (Ingreso Bruto)")
        st.markdown("Visualiza qué combinaciones de sucursal y categoría generan mayor `gross income`.")

        if category_col and "Branch" in filtered_df.columns and "gross income" in filtered_df.columns:
            pivot_df = pd.pivot_table(filtered_df, values="gross income", index="Branch", columns=category_col, aggfunc="sum", fill_value=0)
            fig_heatmap = px.imshow(pivot_df, text_auto=True, aspect="auto", color_continuous_scale="Oranges",
                                    title="Ingreso Bruto por Sucursal y Categoría")
            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.markdown("""
            > 📌 **Interpretación**: Ayuda a entender qué combinaciones generan más ganancia. Las celdas más oscuras indican mayor rentabilidad.
            Permite tomar decisiones de inventario y foco comercial por sucursal.
            """)
        else:
            st.warning("Faltan columnas necesarias para construir el mapa de calor.")

    # Tab 4 - Scatter cogs vs gross income
    with tabs[3]:
        st.subheader("📉 Relación entre Costo y Ganancia Bruta")
        st.markdown("Se analiza la relación entre `cogs` (costos) y `gross income` (ganancia bruta) para cada transacción.")

        if "cogs" in filtered_df.columns and "gross income" in filtered_df.columns:
            fig_scatter = px.scatter(filtered_df, x="cogs", y="gross income", color="Branch" if "Branch" in filtered_df.columns else None,
                                     trendline="ols", title="Costo vs Ganancia Bruta")
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("""
            > 📌 **Interpretación**: Se observa una correlación positiva natural (a mayor costo, mayor ganancia), pero también permite detectar outliers o puntos ineficientes.
            La línea de tendencia ayuda a ver el comportamiento promedio.
            """)
        else:
            st.warning("Las columnas 'cogs' y/o 'gross income' no están disponibles en el conjunto de datos.")

# --- SECCIÓN: ANÁLISIS MULTIVARIADO Y 3D ---
elif menu == "🧬 Análisis Multivariado y 3D":
    st.title("🧬 Análisis Multivariado Avanzado y Visualización 3D")
    
    # Parámetro adicional para esta sección
    n_clusters = st.sidebar.slider("🔢 Número de Clusters (KMeans)", min_value=2, max_value=6, value=3)
    
    # Preparación de datos para PCA
    scaled, numeric = prepare_pca(filtered_df, n_components=3)
    
    tabs = st.tabs(["🔄 Correlación", "📊 Scree Plot", "🧭 PCA 2D", "🌐 PCA 3D", "📌 Segmentos (Clusters)"])

    # Tab 1 - Correlation heatmap
    with tabs[0]:
        st.subheader("🔄 Correlación entre Variables Numéricas")
        st.markdown("Se analiza la relación lineal entre las variables numéricas más relevantes.")
        if not numeric.empty:
            fig_corr, ax = plt.subplots()
            sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
            st.pyplot(fig_corr)
            st.markdown("> 📌 **Interpretación**: Este mapa ayuda a detectar relaciones fuertes como entre `cogs` y `gross income`. Variables muy correlacionadas pueden ser redundantes en modelos predictivos.")
        else:
            st.warning("No hay suficientes variables numéricas para calcular correlaciones.")

    # Tab 2 - Scree Plot
    with tabs[1]:
        st.subheader("📊 Varianza Explicada por Componentes (Scree Plot)")
        st.markdown("El scree plot permite identificar cuántas dimensiones (componentes) son necesarias para representar los datos con poca pérdida de información.")
        if scaled is not None:
            pca_expl = PCA(n_components=min(10, scaled.shape[1]))
            pca_expl.fit(scaled)
            exp_var = pca_expl.explained_variance_ratio_ * 100
            fig_scree = px.bar(x=[f"PC{i+1}" for i in range(len(exp_var))], y=exp_var,
                               labels={'x': 'Componentes', 'y': 'Varianza (%)'},
                               title="Porcentaje de Varianza Explicada por Componentes")
            st.plotly_chart(fig_scree, use_container_width=True)
            st.markdown("> 📌 **Interpretación**: Las primeras dos o tres componentes explican gran parte de la variabilidad, lo que justifica su uso en visualizaciones reducidas.")
        else:
            st.warning("No se pudo calcular la varianza explicada.")

    # Tab 3 - PCA 2D
    with tabs[2]:
        st.subheader("🧭 Análisis PCA en 2D")
        st.markdown("Reducción de dimensionalidad a 2 componentes para observar agrupaciones visuales por categoría.")
        if scaled is not None:
            pca2 = PCA(n_components=2)
            components2 = pca2.fit_transform(scaled)
            pca_df = pd.DataFrame(components2, columns=["PCA1", "PCA2"])
            if category_col:
                pca_df[category_col] = filtered_df[category_col].values[:len(pca_df)]
            fig2d = px.scatter(pca_df, x="PCA1", y="PCA2", color=category_col,
                               title="Proyección PCA 2D por Categoría")
            st.plotly_chart(fig2d, use_container_width=True)
            st.markdown("> 📌 **Interpretación**: Las categorías tienden a agruparse en regiones del espacio, lo cual sugiere diferencias en comportamiento según tipo de producto.")
        else:
            st.warning("No hay suficientes columnas numéricas para aplicar PCA.")

    # Tab 4 - PCA 3D
    with tabs[3]:
        st.subheader("🌐 Visualización PCA en 3D")
        st.markdown("Representación tridimensional de las tres principales componentes principales.")
        if scaled is not None:
            pca3 = PCA(n_components=3).fit_transform(scaled)
            pca3_df = pd.DataFrame(pca3, columns=["PC1", "PC2", "PC3"])
            if "Branch" in filtered_df.columns:
                pca3_df["Branch"] = filtered_df["Branch"].values[:len(pca3_df)]
            fig3d = px.scatter_3d(pca3_df, x="PC1", y="PC2", z="PC3", color="Branch",
                                  title="Proyección 3D por Sucursal")
            st.plotly_chart(fig3d, use_container_width=True)
            st.markdown("> 📌 **Interpretación**: La vista en 3D permite observar agrupaciones y patrones que no son evidentes en 2D. Esto es especialmente útil en segmentos complejos como tiendas o clusters de clientes.")
        else:
            st.warning("No hay suficientes datos para la visualización 3D.")

    # Tab 5 - Clustering
    with tabs[4]:
        st.subheader("📌 Segmentación con KMeans (Clustering)")
        st.markdown("Se agrupan observaciones similares en `k` clusters utilizando los componentes PCA.")
        if scaled is not None:
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit(scaled)
            clusters = kmeans.labels_
            pca2 = PCA(n_components=2)
            comp = pca2.fit_transform(scaled)
            cluster_df = pd.DataFrame(comp, columns=["PCA1", "PCA2"])
            cluster_df["Cluster"] = clusters
            fig_clusters = px.scatter(cluster_df, x="PCA1", y="PCA2", color=cluster_df["Cluster"].astype(str),
                                      title=f"Segmentación de Clientes/Tiendas en {n_clusters} Grupos")
            st.plotly_chart(fig_clusters, use_container_width=True)

            st.markdown("### 🧬 Descripción de Clusters (centroides normalizados)")
            centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric.columns)
            st.dataframe(centers.round(2))

            st.markdown("> 📌 **Interpretación**: Los grupos revelan patrones como clientes que compran mucho pero gastan poco o viceversa. Estos clusters permiten personalizar estrategias comerciales.")
        else:
            st.warning("No es posible aplicar clustering sin datos numéricos suficientes.")

# --- SECCIÓN: ANÁLISIS COMPLEMENTARIOS ---
elif menu == "📌 Análisis Complementarios":
    st.title("📌 Análisis Complementarios y Exploración Específica")
    
    tabs = st.tabs([
        "⭐ Distribución de Calificaciones",
        "💳 Métodos de Pago Preferidos",
        "🧱 Ingreso Bruto por Sucursal y Producto"
    ])

    # Tab 1 - Rating
    with tabs[0]:
        st.subheader("⭐ Distribución de Calificaciones de Clientes")
        st.markdown("Se analiza cómo los clientes calificaron su experiencia en las tiendas.")
        if "Rating" in filtered_df.columns:
            fig = px.histogram(filtered_df, x="Rating", nbins=20, title="Distribución de Calificaciones")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("> 📌 **Interpretación**: Las calificaciones altas indican satisfacción general positiva. Permite monitorear la percepción del servicio.")
        else:
            st.warning("No se encontró la columna 'Rating'.")

    # Tab 2 - Payment
    with tabs[1]:
        st.subheader("💳 Métodos de Pago Preferidos")
        st.markdown("Análisis de frecuencia de los métodos de pago utilizados por los clientes.")
        if "Payment" in filtered_df.columns:
            payment_count = filtered_df["Payment"].value_counts().reset_index()
            payment_count.columns = ["Método", "Cantidad"]
            fig = px.bar(payment_count, x="Método", y="Cantidad", title="Frecuencia de Métodos de Pago")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("> 📌 **Interpretación**: Ayuda a detectar tendencias de medios de pago y planificar infraestructura (como POS, QR, etc).")
        else:
            st.warning("No se encontró la columna 'Payment'.")

    # Tab 3 - Gross Income por Branch y Product line
    with tabs[2]:
        st.subheader("🧱 Ingreso Bruto por Sucursal y Línea de Producto")
        st.markdown("Comparativa del ingreso bruto generado por combinación de tienda y línea de producto.")
        if "gross income" in filtered_df.columns and "Branch" in filtered_df.columns and category_col:
            pivot_df = filtered_df.groupby(["Branch", category_col])["gross income"].sum().reset_index()
            fig = px.sunburst(pivot_df, path=["Branch", category_col], values="gross income",
                              title="Composición de Ingreso Bruto por Sucursal y Línea de Producto")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("> 📌 **Interpretación**: Permite identificar combinaciones altamente rentables que pueden ser reforzadas estratégicamente.")
        else:
            st.warning("Faltan columnas para construir el gráfico.")

# --- SECCIÓN: RESUMEN EJECUTIVO ---
elif menu == "📋 Resumen Ejecutivo":
    st.title("📋 Resumen Ejecutivo de Ventas")
    
    tabs = st.tabs(["📊 Panel de Indicadores", "🧠 Análisis Automatizado"])

    # TAB 1 - Visual
    with tabs[0]:
        st.subheader("📊 Indicadores Clave del Período Seleccionado")

        if "Total" in filtered_df.columns:
            col1, col2, col3 = st.columns(3)
            col1.metric("🧾 Transacciones", len(filtered_df))
            col2.metric("💰 Ventas Totales", f"${filtered_df['Total'].sum():,.2f}")
            col3.metric("🧮 Promedio por Venta", f"${filtered_df['Total'].mean():,.2f}")

            if "Branch" in filtered_df.columns:
                st.markdown("### 🏪 Ventas Totales por Sucursal")
                resumen_branch = filtered_df.groupby("Branch")["Total"].sum().reset_index()
                fig_sucursal = px.bar(resumen_branch, x="Branch", y="Total", title="Ingresos por Sucursal")
                st.plotly_chart(fig_sucursal, use_container_width=True)

            if category_col and "Total" in filtered_df.columns:
                st.markdown("### 📦 Participación por Categoría de Producto")
                fig_categoria = px.pie(filtered_df, names=category_col, values="Total", title="Distribución de Ventas por Categoría")
                st.plotly_chart(fig_categoria, use_container_width=True)

    # TAB 2 - Dinámico
    with tabs[1]:
        st.subheader("🧠 Recomendaciones Gerenciales Automáticas")

        if "Branch" in filtered_df.columns and "Total" in filtered_df.columns:
            resumen_branch = filtered_df.groupby("Branch")["Total"].sum().reset_index()
            avg_ticket = filtered_df.groupby("Branch")["Total"].mean().reset_index()
            top_branch = resumen_branch.sort_values("Total", ascending=False).iloc[0]
            top_avg_ticket = avg_ticket.sort_values("Total", ascending=False).iloc[0]

            st.markdown(f"""
            ### 📊 Hallazgos Clave
            - La sucursal **{top_branch['Branch']}** es la líder en ventas totales con ${top_branch['Total']:,.2f}
            - El ticket promedio más alto se encuentra en la sucursal **{top_avg_ticket['Branch']}** (${top_avg_ticket['Total']:,.2f})
            """)

            if category_col and "Total" in filtered_df.columns:
                cat_total = filtered_df.groupby(category_col)["Total"].sum()
                dominant_cat = cat_total.idxmax()
                dominant_pct = cat_total.max() / cat_total.sum()

                if dominant_pct > 0.4:  # 40% threshold for warning
                    st.warning(f"⚠️ La categoría **{dominant_cat}** representa el {dominant_pct:.0%} del total. Esto podría indicar una **dependencia excesiva** de este tipo de producto.")
                else:
                    st.success(f"La categoría principal es **{dominant_cat}**, con una participación saludable del {dominant_pct:.0%}.")

                st.markdown("### 🔄 Distribución por Categoría")
                st.dataframe(cat_total.reset_index().rename(columns={"Total": "Total Vendido"}))

        st.markdown("## ✅ Recomendaciones de Mejora")
        st.markdown("""
        1. Reforzar promoción en tiendas con menor rendimiento.
        2. Diversificar categorías si existe dependencia alta.
        3. Mantener impulso en tiendas y categorías líderes.
        4. Realizar seguimiento mensual con este dashboard para detectar nuevas oportunidades.

        Este panel genera hallazgos **directamente desde los datos**, facilitando la toma de decisiones en tiempo real.
        """)

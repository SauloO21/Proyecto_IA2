import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.gridspec import GridSpec

# =============================================
# 1. DEFINICIÓN DE DATOS (igual que antes)
# =============================================
data = {
    'Nombre': ['Ana', 'Carlos', 'Daniela', 'Esteban', 'Fernanda', 'Gabriel', 'Hugo', 'Isabel', 'Javier', 'Laura'],
    'Visitas': ['Baja', 'Media', 'Alta', 'Baja', 'Media', 'Alta', 'Baja', 'Media', 'Alta', 'Baja'],
    'Tiempo': ['<3', '3-6', '>6', '<3', '3-6', '>6', '3-6', '>6', '<3', '>6'],
    'Plan': ['Básico', 'Premium', 'Familiar', 'Básico', 'Premium', 'Familiar', 'Básico', 'Premium', 'Familiar', 'Básico'],
    'Edad': ['<30', '<30', '>45', '<30', '<30', '30-45', '>45', '>45', '<30', '30-45'],
    'Quejas': ['Sí', 'No', 'No', 'Sí', 'Sí', 'No', 'Sí', 'No', 'Sí', 'No'],
    'Estado': ['Abandona', 'Continúa', 'Continúa', 'Abandona', 'Continúa', 'Continúa', 'Abandona', 'Continúa', 'Abandona', 'Abandona']
}

df = pd.DataFrame(data)

# =============================================
# 2. FUNCIONES DEL ÁRBOL (igual que antes)
# =============================================
def entropy(p, n):
    total = p + n
    if total == 0: return 0
    q = p / total
    if q == 0 or q == 1: return 0
    return - (q * np.log2(q) + (1 - q) * np.log2(1 - q))

def information_gain(df, attribute, target='Estado'):
    total_p = len(df[df[target] == 'Continúa'])
    total_n = len(df[df[target] == 'Abandona'])
    entropy_total = entropy(total_p, total_n)
    
    values = df[attribute].unique()
    remaining_entropy = 0
    
    for value in values:
        subset = df[df[attribute] == value]
        p_subset = len(subset[subset[target] == 'Continúa'])
        n_subset = len(subset[subset[target] == 'Abandona'])
        weight = len(subset) / len(df)
        remaining_entropy += weight * entropy(p_subset, n_subset)
    
    return entropy_total - remaining_entropy

def build_tree(df, attributes, target='Estado'):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    
    if not attributes:
        return df[target].mode()[0]
    
    best_attribute = max(attributes, key=lambda attr: information_gain(df, attr))
    tree = {best_attribute: {}}
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    
    for value in df[best_attribute].unique():
        subset = df[df[best_attribute] == value]
        if len(subset) == 0:
            tree[best_attribute][value] = df[target].mode()[0]
        else:
            tree[best_attribute][value] = build_tree(subset, remaining_attributes, target)
    
    return tree

# =============================================
# 3. CÁLCULOS Y RESULTADOS (modificado para capturar output)
# =============================================
# Configurar figura antes de los cálculos
plt.figure(figsize=(18, 10))
gs = GridSpec(2, 2, width_ratios=[1.2, 1])

# Área para el árbol (izquierda)
ax_tree = plt.subplot(gs[:, 0])

# Área para la tabla (derecha arriba)
ax_table = plt.subplot(gs[0, 1])
ax_table.axis('off')

# Área para texto (derecha abajo)
ax_text = plt.subplot(gs[1, 1])
ax_text.axis('off')

# --- Mostrar tabla ---
table = ax_table.table(cellText=df.values, 
                      colLabels=df.columns, 
                      loc='center',
                      cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
ax_table.set_title('Datos de Clientes del Gimnasio', pad=20, fontsize=12, fontweight='bold')

# --- Capturar resultados numéricos ---
output_text = []

# Calcular y guardar resultados
p_initial = len(df[df['Estado'] == 'Continúa'])
n_initial = len(df[df['Estado'] == 'Abandona'])
entropy_initial = entropy(p_initial, n_initial)
output_text.append(f"Entropía inicial: {entropy_initial:.4f}\n")

attributes = ['Visitas', 'Tiempo', 'Plan', 'Edad', 'Quejas']
gains = {attr: information_gain(df, attr) for attr in attributes}

output_text.append("Cálculo de entropía, ganancia y restante por atributo:")
for attr in attributes:
    total_p = len(df[df['Estado'] == 'Continúa'])
    total_n = len(df[df['Estado'] == 'Abandona'])
    entropy_total = entropy(total_p, total_n)
    
    values = df[attr].unique()
    remaining_entropy = 0
    
    # Calcular entropía restante sin mostrar valores y pesos
    for value in values:
        subset = df[df[attr] == value]
        p_subset = len(subset[subset['Estado'] == 'Continúa'])
        n_subset = len(subset[subset['Estado'] == 'Abandona'])
        weight = len(subset) / len(df)
        entropy_value = entropy(p_subset, n_subset)
        remaining_entropy += weight * entropy_value
    
    gain = entropy_total - remaining_entropy
    output_text.append(f"\nAtributo: {attr}")
    output_text.append(f"  Entropía restante: {remaining_entropy:.4f}")
    output_text.append(f"  Ganancia de información: {gain:.4f}")

output_text.append("\nGanancia de información por atributo:")
for attr, gain in gains.items():
    output_text.append(f"{attr}: {gain:.4f}")

root_attribute = max(gains, key=gains.get)
output_text.append(f"\nAtributo raíz seleccionado: {root_attribute} (Ganancia: {gains[root_attribute]:.4f})")

# Mostrar texto en el área correspondiente
ax_text.text(0, 1, '\n'.join(output_text), 
            fontfamily='monospace', 
            fontsize=9, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

# =============================================
# 4. VISUALIZACIÓN DEL ÁRBOL (izquierda)
# =============================================
def plot_tree(tree, parent_name='', graph=None, pos=None, level=0, width=4., vert_gap=0.8, xcenter=0.5, edge_labels=None, parent_value=None):
    """
    Función para construir y asignar posiciones a los nodos del árbol,
    y añadir etiquetas a las ramas.
    """
    if graph is None:
        graph = nx.DiGraph()
        pos = {}
        edge_labels = {}

    if not isinstance(tree, dict):
        # Nodo hoja: incluir el valor del atributo del que proviene
        label = f"{tree} ({parent_value})" if parent_value else str(tree)
        graph.add_node(parent_name, label=label)
        pos[parent_name] = (xcenter, -level * vert_gap)
        return graph, pos, edge_labels

    # Nodo interno
    root = list(tree.keys())[0]
    if parent_name:
        graph.add_edge(parent_name, root)
    graph.add_node(root, label=root)
    pos[root] = (xcenter, -level * vert_gap)

    # Dividir espacio horizontal entre los hijos
    children = tree[root]
    num_children = len(children)
    step = width / max(1, num_children)
    next_x = xcenter - width / 2 + step / 2

    for value, subtree in children.items():
        # Generar un nombre único para el nodo hijo
        child_name = f"{root}_{value}_{level}"
        graph.add_node(child_name, label=value)
        pos[child_name] = (next_x, -(level + 0.5) * vert_gap)  # Asignar posición al nodo hijo
        graph.add_edge(root, child_name)
        edge_labels[(root, child_name)] = value  # Añadir etiqueta a la rama

        # Recursión para procesar el subárbol
        graph, pos, edge_labels = plot_tree(subtree, child_name, graph, pos, level + 1, width / num_children, vert_gap, next_x, edge_labels, parent_value=f"{root}={value}")
        next_x += step  # Mover al siguiente nodo en el espacio horizontal

    return graph, pos, edge_labels

# Construir y dibujar árbol
tree = build_tree(df, attributes)
graph, pos, edge_labels = plot_tree(tree)

# Obtener etiquetas de nodos y colores
node_labels = nx.get_node_attributes(graph, 'label')
node_colors = ['skyblue' if not label.startswith(('Continúa', 'Abandona')) 
               else 'lightgreen' if label.startswith('Continúa') 
               else 'salmon' for label in node_labels.values()]

# Dibujar nodos y aristas
nx.draw(graph, pos, 
        with_labels=True, 
        labels=node_labels,
        node_size=3000,
        node_color=node_colors,
        font_size=7,  # Reducir tamaño de la fuente de los nodos
        font_weight='bold',
        edge_color='gray',
        ax=ax_tree)

# Dibujar etiquetas en las ramas
nx.draw_networkx_edge_labels(graph, pos, 
                             edge_labels=edge_labels, 
                             font_size=7,  # Reducir tamaño de la fuente de las ramas
                             font_color='black')

# Configurar título del árbol
ax_tree.set_title('Árbol de Decisión - Gimnasio', pad=20, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
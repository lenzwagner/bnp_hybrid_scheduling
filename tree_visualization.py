import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List, Tuple
import math


class BnPTreeVisualizer:
    """
    Visualize Branch-and-Price search tree with color-coded nodes.

    Features:
    - Hierarchical tree layout
    - Color coding by node status and fathoming reason
    - Branching information displayed on edges
    - Interactive and static export options
    - Customizable styling
    """

    # Color scheme for different node types
    COLORS = {
        'root': '#4A90E2',  # Blue
        'open': '#F5F5F5',  # Light gray
        'branched': '#FFD700',  # Gold
        'integral': '#50C878',  # Emerald green
        'fathomed_bound': '#FF6B6B',  # Red
        'fathomed_infeasible': '#95A5A6',  # Gray
        'fathomed_integral': '#2ECC71',  # Green
        'fathomed_error': '#34495E',  # Dark gray
        'fathomed_bound_after_heuristic': '#E74C3C'  # Darker red
    }

    # Node shapes
    SHAPES = {
        'root': 's',  # square
        'open': 'o',  # circle
        'branched': 'o',  # circle
        'fathomed': 'X',  # X mark
        'integral': 'D'  # diamond
    }

    def __init__(self, bnp_solver):
        """
        Initialize visualizer with BnP solver.

        Args:
            bnp_solver: BranchAndPrice instance after solve()
        """
        self.bnp_solver = bnp_solver
        self.G = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """Build NetworkX graph from BnP tree."""
        for node_id, node in self.bnp_solver.nodes.items():
            # Add node with attributes
            self.G.add_node(
                node_id,
                status=node.status,
                depth=node.depth,
                lp_bound=node.lp_bound,
                is_integral=node.is_integral,
                fathom_reason=node.fathom_reason,
                path=node.path,
                branching_info=self._get_branching_info(node)
            )

            # Add edge from parent
            if node.parent_id is not None:
                parent = self.bnp_solver.nodes[node.parent_id]
                edge_label = self._get_edge_label(parent, node)
                self.G.add_edge(node.parent_id, node_id, label=edge_label)

    def _get_branching_info(self, node) -> str:
        """Extract branching information from node's constraints."""
        if not node.branching_constraints:
            return "Root"

        # Get the last branching constraint (most recent)
        last_constraint = node.branching_constraints[-1]

        from branching_constraints import MPVariableBranching, SPVariableBranching

        if isinstance(last_constraint, MPVariableBranching):
            direction_symbol = "≤" if last_constraint.direction == 'left' else "≥"
            return f"λ[{last_constraint.profile},{last_constraint.column}] {direction_symbol} {last_constraint.bound}"

        elif isinstance(last_constraint, SPVariableBranching):
            return f"x[{last_constraint.profile},{last_constraint.agent},{last_constraint.period}] = {last_constraint.value}"

        return "Unknown"

    def _get_edge_label(self, parent, child) -> str:
        """Create edge label showing branching decision."""
        if not child.branching_constraints:
            return ""

        last_constraint = child.branching_constraints[-1]

        from branching_constraints import MPVariableBranching, SPVariableBranching

        if isinstance(last_constraint, MPVariableBranching):
            if last_constraint.direction == 'left':
                return f"λ ≤ {last_constraint.bound}"
            else:
                return f"λ ≥ {last_constraint.bound}"

        elif isinstance(last_constraint, SPVariableBranching):
            return f"x = {last_constraint.value}"

        return ""

    def _get_node_color(self, node_data) -> str:
        """Determine node color based on status."""
        status = node_data['status']
        fathom_reason = node_data.get('fathom_reason')

        if status == 'open':
            return self.COLORS['open']
        elif status == 'branched':
            return self.COLORS['branched']
        elif status == 'fathomed':
            if fathom_reason == 'integral':
                return self.COLORS['fathomed_integral']
            elif fathom_reason == 'bound':
                return self.COLORS['fathomed_bound']
            elif fathom_reason == 'bound_after_heuristic':
                return self.COLORS['fathomed_bound_after_heuristic']
            elif fathom_reason == 'infeasible':
                return self.COLORS['fathomed_infeasible']
            elif fathom_reason == 'error':
                return self.COLORS['fathomed_error']
            else:
                return self.COLORS['fathomed_bound']  # Default
        elif status == 'integral':
            return self.COLORS['integral']

        # Root node
        if node_data.get('depth') == 0:
            return self.COLORS['root']

        return self.COLORS['open']

    def _hierarchical_layout(self) -> Dict[int, Tuple[float, float]]:
        """
        Create hierarchical tree layout.

        Returns:
            Dictionary mapping node_id to (x, y) position
        """
        # Get positions by depth
        pos = {}
        depth_counts = {}  # Count nodes at each depth
        depth_current = {}  # Current position counter at each depth

        # Initialize counters
        for node_id in self.G.nodes():
            depth = self.G.nodes[node_id]['depth']
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            depth_current[depth] = 0

        # Assign positions
        max_depth = max(depth_counts.keys())

        for node_id in sorted(self.G.nodes(), key=lambda n: (self.G.nodes[n]['depth'], n)):
            depth = self.G.nodes[node_id]['depth']

            # Y position (negative so root is at top)
            y = -depth

            # X position (spread nodes horizontally)
            if depth_counts[depth] == 1:
                x = 0
            else:
                # Spread nodes evenly
                width = depth_counts[depth] - 1
                x = (depth_current[depth] - width / 2) * (10 / max(1, max_depth))

            pos[node_id] = (x, y)
            depth_current[depth] += 1

        return pos

    def _radial_layout(self) -> Dict[int, Tuple[float, float]]:
        """
        Create radial tree layout (root at center).

        Returns:
            Dictionary mapping node_id to (x, y) position
        """
        pos = {}

        # Get children for each node
        children = {node: list(self.G.successors(node)) for node in self.G.nodes()}

        def place_nodes(node, angle_start, angle_end, radius):
            """Recursively place nodes in radial pattern."""
            # Place current node
            if node == 0:  # Root at center
                pos[node] = (0, 0)
            else:
                angle = (angle_start + angle_end) / 2
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pos[node] = (x, y)

            # Place children
            child_list = children[node]
            if child_list:
                angle_range = angle_end - angle_start
                angle_per_child = angle_range / len(child_list)

                for i, child in enumerate(child_list):
                    child_start = angle_start + i * angle_per_child
                    child_end = child_start + angle_per_child
                    place_nodes(child, child_start, child_end, radius + 1)

        # Start from root
        place_nodes(0, 0, 2 * math.pi, 1)

        return pos

    def plot(self, layout='hierarchical', figsize=(20, 12), show_bounds=True,
             show_edge_labels=True, save_path=None, dpi=300):
        """
        Create and display the tree visualization.

        Args:
            layout: 'hierarchical' or 'radial'
            figsize: Figure size (width, height) in inches
            show_bounds: Show LP bounds on nodes
            show_edge_labels: Show branching decisions on edges
            save_path: If provided, save figure to this path
            dpi: Resolution for saved figure
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        # Get layout
        if layout == 'radial':
            pos = self._radial_layout()
        else:
            pos = self._hierarchical_layout()

        # Prepare node colors and labels
        node_colors = []
        node_labels = {}

        for node_id in self.G.nodes():
            node_data = self.G.nodes[node_id]
            node_colors.append(self._get_node_color(node_data))

            # Create label
            label_parts = [f"Node {node_id}"]

            if show_bounds and node_data['lp_bound'] < float('inf'):
                label_parts.append(f"LP: {node_data['lp_bound']:.2f}")

            if node_data.get('fathom_reason'):
                reason_short = {
                    'integral': 'INT',
                    'bound': 'BND',
                    'bound_after_heuristic': 'BND(H)',
                    'infeasible': 'INF',
                    'error': 'ERR'
                }
                short = reason_short.get(node_data['fathom_reason'], node_data['fathom_reason'][:3].upper())
                label_parts.append(f"[{short}]")

            node_labels[node_id] = "\n".join(label_parts)

        # Draw edges
        nx.draw_networkx_edges(
            self.G, pos, ax=ax,
            edge_color='#34495E',
            width=2,
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->'
        )

        # Draw edge labels (branching decisions)
        if show_edge_labels:
            edge_labels = nx.get_edge_attributes(self.G, 'label')
            nx.draw_networkx_edge_labels(
                self.G, pos,
                edge_labels=edge_labels,
                font_size=8,
                font_color='#2C3E50',
                ax=ax
            )

        # Draw nodes
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            node_color=node_colors,
            node_size=3000,
            node_shape='o',
            edgecolors='#2C3E50',
            linewidths=2
        )

        # Draw labels
        nx.draw_networkx_labels(
            self.G, pos,
            labels=node_labels,
            font_size=9,
            font_weight='bold',
            font_color='#2C3E50',
            ax=ax
        )

        # Create legend
        legend_elements = [
            mpatches.Patch(color=self.COLORS['root'], label='Root Node'),
            mpatches.Patch(color=self.COLORS['open'], label='Open'),
            mpatches.Patch(color=self.COLORS['branched'], label='Branched'),
            mpatches.Patch(color=self.COLORS['fathomed_integral'], label='Fathomed: Integral'),
            mpatches.Patch(color=self.COLORS['fathomed_bound'], label='Fathomed: Bound'),
            mpatches.Patch(color=self.COLORS['fathomed_bound_after_heuristic'], label='Fathomed: Bound (Heuristic)'),
            mpatches.Patch(color=self.COLORS['fathomed_infeasible'], label='Fathomed: Infeasible'),
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True
        )

        # Add title with statistics
        title = f"Branch-and-Price Search Tree\n"
        title += f"Strategy: {self.bnp_solver.branching_strategy.upper()} | "
        title += f"Nodes: {len(self.G.nodes())} | "
        title += f"Fathomed: {self.bnp_solver.stats['nodes_fathomed']} | "
        title += f"Gap: {self.bnp_solver.gap:.2%}" if self.bnp_solver.gap < float('inf') else f"Gap: ∞"

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"\n✅ Tree visualization saved to: {save_path}")

        plt.show()

    def plot_detailed(self, figsize=(24, 16), save_path=None):
        """
        Create detailed visualization with more information.

        Shows:
        - Full branching constraints
        - Path from root
        - Detailed bounds
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        pos = self._hierarchical_layout()

        # Node colors and detailed labels
        node_colors = []
        node_labels = {}

        for node_id in self.G.nodes():
            node_data = self.G.nodes[node_id]
            node_colors.append(self._get_node_color(node_data))

            # Detailed label
            label_parts = [
                f"Node {node_id}",
                f"Path: {node_data['path'] if node_data['path'] else 'root'}",
            ]

            if node_data['lp_bound'] < float('inf'):
                label_parts.append(f"LP: {node_data['lp_bound']:.4f}")

            branching_info = node_data.get('branching_info', '')
            if branching_info and branching_info != "Root":
                label_parts.append(f"{branching_info}")

            if node_data.get('fathom_reason'):
                label_parts.append(f"Status: {node_data['fathom_reason'].upper()}")

            node_labels[node_id] = "\n".join(label_parts)

        # Draw edges
        nx.draw_networkx_edges(
            self.G, pos, ax=ax,
            edge_color='#34495E',
            width=2.5,
            alpha=0.6,
            arrows=True,
            arrowsize=25,
            arrowstyle='-|>'
        )

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.G, 'label')
        nx.draw_networkx_edge_labels(
            self.G, pos,
            edge_labels=edge_labels,
            font_size=10,
            font_color='#E74C3C',
            font_weight='bold',
            ax=ax
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            node_color=node_colors,
            node_size=5000,
            node_shape='o',
            edgecolors='#2C3E50',
            linewidths=3
        )

        # Draw labels
        nx.draw_networkx_labels(
            self.G, pos,
            labels=node_labels,
            font_size=8,
            font_weight='bold',
            font_color='#2C3E50',
            ax=ax
        )

        # Legend
        legend_elements = [
            mpatches.Patch(color=self.COLORS['root'], label='Root Node'),
            mpatches.Patch(color=self.COLORS['open'], label='Open'),
            mpatches.Patch(color=self.COLORS['branched'], label='Branched'),
            mpatches.Patch(color=self.COLORS['fathomed_integral'], label='Fathomed: Integral Solution'),
            mpatches.Patch(color=self.COLORS['fathomed_bound'], label='Fathomed: Pruned by Bound'),
            mpatches.Patch(color=self.COLORS['fathomed_bound_after_heuristic'],
                           label='Fathomed: Pruned by Bound (IP Heuristic)'),
            mpatches.Patch(color=self.COLORS['fathomed_infeasible'], label='Fathomed: Infeasible'),
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True
        )

        # Title with comprehensive statistics
        stats = self.bnp_solver.stats
        title = f"Branch-and-Price Search Tree (Detailed)\n"
        title += f"Strategy: {self.bnp_solver.branching_strategy.upper()} | "
        title += f"Total Nodes: {stats['nodes_explored']} | "
        title += f"Fathomed: {stats['nodes_fathomed']} | "
        title += f"Branched: {stats['nodes_branched']}\n"

        if self.bnp_solver.incumbent < float('inf'):
            title += f"Incumbent (UB): {self.bnp_solver.incumbent:.4f} | "
        else:
            title += f"Incumbent (UB): None | "

        title += f"Best LB: {self.bnp_solver.best_lp_bound:.4f} | "
        title += f"Gap: {self.bnp_solver.gap:.2%}" if self.bnp_solver.gap < float('inf') else f"Gap: ∞"

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n✅ Detailed tree visualization saved to: {save_path}")

        plt.show()

    def export_to_graphviz(self, filename='bnp_tree.dot'):
        """
        Export tree to Graphviz DOT format for external visualization.

        Args:
            filename: Output filename for DOT file
        """
        with open(filename, 'w') as f:
            f.write("digraph BnPTree {\n")
            f.write("    rankdir=TB;\n")
            f.write("    node [shape=box, style=filled];\n\n")

            # Write nodes
            for node_id in self.G.nodes():
                node_data = self.G.nodes[node_id]
                color = self._get_node_color(node_data).replace('#', '')

                label = f"Node {node_id}\\n"
                if node_data['lp_bound'] < float('inf'):
                    label += f"LP: {node_data['lp_bound']:.2f}\\n"
                if node_data.get('fathom_reason'):
                    label += f"[{node_data['fathom_reason']}]"

                f.write(f'    {node_id} [label="{label}", fillcolor="#{color}"];\n')

            f.write("\n")

            # Write edges
            for edge in self.G.edges():
                edge_label = self.G.edges[edge].get('label', '')
                f.write(f'    {edge[0]} -> {edge[1]} [label="{edge_label}"];\n')

            f.write("}\n")

        print(f"\n✅ Graphviz DOT file saved to: {filename}")
        print(f"   Render with: dot -Tpng {filename} -o tree.png")

    def print_tree_statistics(self):
        """Print detailed statistics about the search tree."""
        print("\n" + "=" * 80)
        print(" SEARCH TREE STATISTICS ".center(80, "="))
        print("=" * 80)

        # Count nodes by status
        status_counts = {}
        fathom_reason_counts = {}
        depth_counts = {}

        for node_id in self.G.nodes():
            node_data = self.G.nodes[node_id]

            status = node_data['status']
            status_counts[status] = status_counts.get(status, 0) + 1

            if node_data.get('fathom_reason'):
                reason = node_data['fathom_reason']
                fathom_reason_counts[reason] = fathom_reason_counts.get(reason, 0) + 1

            depth = node_data['depth']
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        print("\nNode Status Distribution:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status.capitalize():20}: {count:4d} ({count / len(self.G.nodes()) * 100:5.1f}%)")

        if fathom_reason_counts:
            print("\nFathoming Reasons:")
            for reason, count in sorted(fathom_reason_counts.items()):
                print(f"  {reason.replace('_', ' ').capitalize():30}: {count:4d}")

        print("\nTree Depth Distribution:")
        for depth in sorted(depth_counts.keys()):
            count = depth_counts[depth]
            bar = "█" * int(count * 40 / max(depth_counts.values()))
            print(f"  Depth {depth:2d}: {count:4d} {bar}")

        print(f"\nTree Metrics:")
        print(f"  Total Nodes:        {len(self.G.nodes())}")
        print(f"  Maximum Depth:      {max(depth_counts.keys())}")
        print(f"  Average Depth:      {sum(d * c for d, c in depth_counts.items()) / len(self.G.nodes()):.2f}")
        print(f"  Leaf Nodes:         {sum(1 for n in self.G.nodes() if self.G.out_degree(n) == 0)}")
        print(
            f"  Branch Factor (avg): {sum(self.G.out_degree(n) for n in self.G.nodes()) / max(1, len(self.G.nodes()) - sum(1 for n in self.G.nodes() if self.G.out_degree(n) == 0)):.2f}")

        print("=" * 80 + "\n")

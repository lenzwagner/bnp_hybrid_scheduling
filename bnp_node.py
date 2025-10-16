import math

class BnPNode:
    """
    Represents a node in the Branch-and-Price search tree.

    This class stores all information that a node in the B&P tree requires:
    - Position in tree (ID, parent, depth)
    - LP bound and status
    - Branching constraints from root to this node
    - Column pool (available columns at this node)
    - Solutions (LP and IP)

    Attributes:
        node_id: Unique node ID
        parent_id: Parent node ID (None for root)
        depth: Depth in search tree (root = 0)
        lp_bound: LP relaxation value at this node
        is_integral: Boolean - is the LP solution integral?
        branching_constraints: List of all branching decisions on path from root
        column_pool: Dict of available columns {(profile, col_id): col_data}
        status: 'open', 'fathomed', 'branched', 'integral'
        fathom_reason: Reason for fathoming (if status='fathomed')
    """

    def __init__(self, node_id, parent_id=None, depth=0, path=''):
        # Tree position
        self.node_id = node_id
        self.parent_id = parent_id
        self.depth = depth
        self.path = path

        # Bounds and status
        self.lp_bound = float('inf')
        self.is_integral = False

        # Branching constraints
        self.branching_constraints = []

        # Column management
        self.column_pool = {}

        # Status tracking
        self.status = 'open'  # 'open', 'fathomed', 'branched', 'integral'
        self.fathom_reason = None  # 'integral', 'bound', 'infeasible'

        # Solutions
        self.master_solution = None  # LP solution (dict with variable values)
        self.most_fractional_var = None  # Info about most fractional variable

    def __repr__(self):
        """String representation for debugging."""
        return (f"Node(id={self.node_id}, depth={self.depth}, "
                f"status={self.status}, bound={self.lp_bound:.2f}, "
                f"parent={self.parent_id})")

    def __str__(self):
        """Detailed string representation."""
        info = [
            f"Node {self.node_id}:",
            f"  Depth: {self.depth}",
            f"  Parent: {self.parent_id}",
            f"  Status: {self.status}",
            f"  LP Bound: {self.lp_bound:.6f}",
            f"  Is Integral: {self.is_integral}",
            f"  Branching Constraints: {len(self.branching_constraints)}",
            f"  Columns in Pool: {len(self.column_pool)}"
        ]

        if self.fathom_reason:
            info.append(f"  Fathom Reason: {self.fathom_reason}")
        return "\n".join(info)

    def __repr__(self):
        """String representation for debugging."""
        path_str = f"'{self.path}'" if self.path else "'root'"
        return (f"Node(id={self.node_id}, path={path_str}, depth={self.depth}, "
                f"status={self.status}, bound={self.lp_bound:.2f})")
use std::{cell::RefCell, fmt::Debug, rc::Rc};

type Link<T, U> = Rc<RefCell<Node<T, U>>>;

#[derive(Debug)]
struct NodeStats {
    visits: u32,
    value: f32,
}

struct Node<T, U> {
    parent: Option<Link<T, U>>,
    children: Vec<Link<T, U>>,
    stats: NodeStats,
    state: T,
    action: Option<U>,
}

impl<T, U> Node<T, U> {
    fn new(state: T, action: Option<U>) -> Link<T, U> {
        Rc::new(RefCell::new(Self {
            parent: None,
            children: Vec::new(),
            stats: NodeStats {
                visits: 0,
                value: 0.0,
            },
            state,
            action: action,
        }))
    }
}

impl<T, U> Debug for Node<T, U>
where
    T: Debug,
    U: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("children", &self.children)
            .field("stats", &self.stats)
            .field("state", &self.state)
            .field("action", &self.action)
            .finish()
    }
}

#[derive(Debug)]
struct Tree<T, U> {
    root: Link<T, U>,
}

impl<T, U> Tree<T, U> {
    fn new(state: T) -> Self {
        Self {
            root: Node::new(state, None),
        }
    }

    fn add_child(&mut self, node: &Link<T, U>, state: T, action: U) {
        let new_node = Node::new(state, Some(action));
        new_node.borrow_mut().parent = Some(node.clone());
        node.borrow_mut().children.push(new_node);
    }
}

trait GameState<U> {
    fn get_actions(&self) -> Vec<U>;
    fn get_next_state(&self, action: U) -> Self;
    fn is_terminal(&self) -> Option<f32>;
}

struct MCTS<T: GameState<U>, U> {
    tree: Tree<T, U>,
}

impl<T, U> MCTS<T, U>
where
    T: GameState<U> + Clone + Debug,
    U: Copy + Debug,
{
    fn new(state: T) -> Self {
        Self {
            tree: Tree::new(state),
        }
    }

    fn search(&mut self, iterations: u32) {
        for _ in 0..iterations {
            let mut node = self.tree.root.clone();
            let mut state = node.borrow().state.clone();

            while !node.borrow().children.is_empty() && !state.is_terminal().is_some() {
                node = self.select(&node);
                state = node.borrow().state.clone();
            }

            if let Some(value) = state.is_terminal() {
                self.backpropagate(&node, value);
            } else {
                self.expand(&node);
                let node = self.select(&node);
                let value = self.simulate(&node.borrow().state);
                self.backpropagate(&node, value);
            }
        }
    }

    fn expand(&mut self, node: &Link<T, U>) {
        let state = node.borrow().state.clone();

        for action in state.get_actions() {
            let next_state = state.get_next_state(action);
            self.tree.add_child(node, next_state, action);
        }
    }

    fn select(&self, node: &Link<T, U>) -> Link<T, U> {
        node.borrow()
            .children
            .iter()
            .max_by_key(|child| self.ucb1(child) as i32)
            .unwrap()
            .clone()
    }

    fn ucb1(&self, node: &Link<T, U>) -> f32 {
        let parent_visits = node.borrow().parent.clone().unwrap().borrow().stats.visits;
        let node_visits = node.borrow().stats.visits;
        let node_value = node.borrow().stats.value;

        // avg_value + sqrt(2 * ln(parent_visits) / node_visits)
        node_value / node_visits as f32
            + (2.0 * (parent_visits as f32).ln() / node_visits as f32).sqrt()
    }

    fn simulate(&self, state: &T) -> f32 {
        let mut state = state.clone();

        while !state.is_terminal().is_some() {
            let actions = state.get_actions();
            let action = actions[rand::random::<usize>() % actions.len()];
            state = state.get_next_state(action);
        }

        state.is_terminal().unwrap()
    }

    fn backpropagate(&mut self, node: &Link<T, U>, value: f32) {
        let mut current = node.clone();

        loop {
            current.borrow_mut().stats.visits += 1;
            current.borrow_mut().stats.value += value;

            let parent = current.borrow().parent.clone();

            match parent {
                Some(parent) => current = parent,
                None => break,
            }
        }
    }

    fn get_principal_variation(&self) -> Vec<U> {
        let mut node = self.tree.root.clone();
        let mut actions = Vec::new();

        loop {
            let mut child = node.borrow().children.clone();
            let child = child.iter().max_by_key(|child| child.borrow().stats.visits);

            if let Some(child) = child {
                actions.push(child.borrow().action.unwrap());
                node = child.clone();
            } else {
                break;
            }
        }

        actions
    }
}

mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct NimState {
        heap: u32,
    }

    impl NimState {
        fn new(heap: u32) -> Self {
            Self { heap }
        }
    }

    impl GameState<u32> for NimState {
        fn get_actions(&self) -> Vec<u32> {
            let max = if self.heap > 11 { 10 } else { self.heap };
            (1..=max).collect()
        }

        fn get_next_state(&self, action: u32) -> Self {
            Self::new(self.heap - action)
        }

        fn is_terminal(&self) -> Option<f32> {
            if self.heap <= 0 {
                Some(1.0)
            } else {
                None
            }
        }
    }

    #[test]
    fn test_tree() {
        let mut tree = Tree::new(0);
        let root = tree.root.clone();
        tree.add_child(&root, 1, 1);
        tree.add_child(&root, 2, 2);
        let root = tree.root.borrow();

        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0].borrow().state, 1);
        assert_eq!(root.children[1].borrow().state, 2);
    }

    #[test]
    fn test_mcts() {
        // this test is not deterministic, but more for a sanity/visual check

        let mut mcts = MCTS::new(NimState::new(100));
        mcts.search(1);

        for child in mcts.tree.root.borrow().children.iter() {
            println!("{:?}", child.borrow().stats);
        }

        mcts.search(1000);

        println!("Principal variation: {:?}", mcts.get_principal_variation());
    }
}

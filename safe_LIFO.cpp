#include <iostream>

template<typename T>
class SafeStack {
private:
    int m_size;      // Maximum size
    int m_top;       // Current top index
    T* m_stack;      // Array to store elements

public:
    SafeStack(int size = 0) : m_size{size}, m_top{-1} {
        m_stack = new T[size];
    }

    ~SafeStack() {
        delete[] m_stack;
    }

    // Push element onto stack
    bool push(const T& element) {
        if (m_top >= m_size - 1) {
            throw std::out_of_range("Stack overflow");
            return false;
        }
        m_stack[++m_top] = element;
        return true;
    }

    // Pop element from stack
    T pop() {
        if (m_top < 0) {
            throw std::out_of_range("Stack underflow");
        }
        return m_stack[m_top--];
    }

    // Peek at top element
    T peek() const {
        if (m_top < 0) {
            throw std::out_of_range("Stack is empty");
        }
        return m_stack[m_top];
    }

    // Check if stack is empty
    bool isEmpty() const {
        return m_top < 0;
    }

    // Check if stack is full
    bool isFull() const {
        return m_top >= m_size - 1;
    }

    // Get current size
    int getCurrentSize() const {
        return m_top + 1;
    }
};


int main() {
    SafeStack<int> stack(5);  // Create stack of size 5

    try {
        // Push some elements
        stack.push(10);
        stack.push(20);
        stack.push(30);

        // Peek at top element
        std::cout << "Top element: " << stack.peek() << std::endl;  // 30

        // Pop elements
        std::cout << "Popped: " << stack.pop() << std::endl;  // 30
        std::cout << "Popped: " << stack.pop() << std::endl;  // 20

        // Check current size
        std::cout << "Current size: " << stack.getCurrentSize() << std::endl;  // 1

        // Check if empty
        std::cout << "Is empty? " << (stack.isEmpty() ? "Yes" : "No") << std::endl;
    }
    catch (const std::out_of_range& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    return 0;

}




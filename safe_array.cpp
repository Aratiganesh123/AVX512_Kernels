#include<iostream>


#include <iostream>

template <typename T>
class SafeArray {
public:
    SafeArray(int size = 0) : m_size{size}, m_arr{new T[size]} 
    {
    }

    ~SafeArray() {  // Added destructor to prevent memory leak
        delete[] m_arr;
    }

    int getSize() const {
        return m_size;
    }

    T& operator[](int i) {
        if (static_cast<unsigned int>(i) >= m_size) {  // Changed > to >=
            throw std::out_of_range("Index out of bounds");  // Fixed throw syntax
        }
        return m_arr[i];
    }

private:
    T* m_arr;
    int m_size;
};



int main() {
    SafeArray<int> arr(5);  // Create array of size 5
    
    // Valid operations
    arr[0] = 10;
    arr[4] = 50;
    
    try {
        arr[5] = 60;  // This will throw an exception
    } catch (const std::out_of_range& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
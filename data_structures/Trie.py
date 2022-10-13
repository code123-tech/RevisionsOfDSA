class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEnd = False
        self.count = 0


class Trie:
    def __init__(self):
        self.__root = TrieNode()

    def insert(self, word):
        node = self.__root

        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]

        node.count = 1 if node.isEnd == False else node.count+1
        node.isEnd = True

    def search(self, word):
        node = self.__root

        for ch in word:
            if ch not in node.children:
                return False

            node = node.children[ch]

        return True

    def countPrefixesInString(self, s):
        node = self.__root

        prefix_count = 0

        for ch in s:
            if ch not in node.children:
                break

            node = node.children[ch]
            prefix_count += node.count if node.isEnd else 0

        return prefix_count

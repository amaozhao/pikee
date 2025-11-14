"""文档切分相关的 LLM 提示词模板.

所有提示词使用英文以获得更好的 LLM 性能。
"""

from pikee.pipeline.models.document import Document


class ChunkingPrompts:
    """文档切分提示词管理器."""

    @staticmethod
    def build_first_chunk_summary(content: str, document: Document) -> str:
        """构建第一个 chunk 摘要提示.

        Args:
            content: Chunk 内容
            document: 文档对象

        Returns:
            str: 提示文本
        """
        return f"""# Document Source

Title: {document.title}
Path: {document.file_path}

# Text Content

{content}

# Task

Generate a concise summary of the above text, capturing its main content.

# Output

Output only the summary content, without any additional information.
"""

    @staticmethod
    def build_last_chunk_summary(chunk: str, prev_summary: str, document: Document) -> str:
        """构建最后 chunk 摘要提示.

        Args:
            chunk: Chunk 内容
            prev_summary: 前一个 chunk 的摘要
            document: 文档对象

        Returns:
            str: 提示文本
        """
        return f"""# Document Source

Title: {document.title}

# Previous Context Summary

{prev_summary}

# Current Text

{chunk}

# Task

Generate a summary for the current text, considering the previous context summary.

# Output

Output only the summary content, without any additional information.
"""

    @staticmethod
    def build_resplit(content: str, summary: str, document: Document) -> str:
        """构建重切分提示.

        Args:
            content: 待切分的内容
            summary: 前文摘要
            document: 文档对象

        Returns:
            str: 提示文本
        """
        lines = content.split("\n")
        numbered_content = "\n".join([f"{i + 1}| {line}" for i, line in enumerate(lines)])

        return f"""# Document Source

Title: {document.title}

# Previous Context Summary

{summary}

# Text to Split

{numbered_content}

# Task

1. Understand the previous context summary and the text to split
2. Analyze the text structure and split it into "Part 1" and "Part 2", without missing any content
3. Provide the ending line number of "Part 1" (starting from 1)
4. Generate a summary for "Part 1"
5. Generate a summary for "Part 2" (considering the context)

# Output Format

Follow this format strictly:

Thinking: [Analyze the text structure and explain how to split]

<result>
<chunk>
  <endline>[Ending line number of Part 1]</endline>
  <summary>[Summary of Part 1]</summary>
</chunk>
<chunk>
  <summary>[Summary of Part 2]</summary>
</chunk>
</result>
"""


class AtomExtractionPrompts:
    """Atom 提取提示词管理器."""

    @staticmethod
    def build_atom_extraction(content: str, chunk_id: str = "", title: str = "") -> str:
        """构建 Atom 提取提示.

        Args:
            content: Chunk 内容
            chunk_id: Chunk ID
            title: Chunk 标题

        Returns:
            str: 提示文本
        """
        return f"""# Task

You are an expert at analyzing text and extracting atomic questions that can be answered by the given content.

# Source Information

Chunk ID: {chunk_id}
Title: {title}

# Content

{content}

# Instructions

Extract atomic questions from the above content. Each atomic question should:

1. **Be a complete, standalone question** that can be answered directly by the content
2. **Include necessary entities** (don't use pronouns like "he", "it", "this")
3. **Be independent** (understandable without additional context)
4. **Cover different aspects** (who, what, when, where, why, how)

# Output Format

Return ONLY a JSON array of questions (no explanation):

["Question 1?", "Question 2?", "Question 3?", ...]

# Example

Content: "Parasite is a 2019 South Korean black comedy thriller film directed by Bong Joon-ho. 
The film won Best Picture at the 92nd Academy Awards in 2020, making history as the first 
non-English language film to win the award."

Output:
["Who directed the movie Parasite?", "What genre is the movie Parasite?", "When was the movie Parasite released?", "What award did Parasite win at the 2020 Academy Awards?", "What is significant about Parasite winning Best Picture?"]

Now extract questions from the provided content above.
"""


def get_chunking_prompts() -> ChunkingPrompts:
    """获取文档切分提示词管理器实例.

    Returns:
        ChunkingPrompts: 提示词管理器实例
    """
    return ChunkingPrompts()


def get_atom_extraction_prompts() -> AtomExtractionPrompts:
    """获取 Atom 提取提示词管理器实例.

    Returns:
        AtomExtractionPrompts: 提示词管理器实例
    """
    return AtomExtractionPrompts()

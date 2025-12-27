"""Tool schema generation and documentation for prompts."""

from __future__ import annotations

from typing import Any

from beartype import beartype

from radiant_harness.tools.tool import Tool

# Valid JSON Schema types for tool parameters
VALID_PARAM_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}


@beartype
class ToolDocumenter:
    """Generates tool schemas and documentation.

    Handles:
    - OpenAI-compatible tool schema generation
    - Prompt documentation formatting
    - Tool categorization and filtering
    - Schema validation

    Example:
        documenter = ToolDocumenter(tools=[zoom_tool, crop_tool])
        schemas = documenter.get_schemas()
        docs = documenter.generate_prompt_documentation()
    """

    @beartype
    def __init__(self, tools: list[Tool] | None = None) -> None:
        """Initialize tool documenter.

        Args:
            tools: List of tools to document. Can be empty and tools added later.
        """
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    @beartype
    def register(self, tool: Tool) -> None:
        """Register a tool for documentation.

        Args:
            tool: Tool to register
        """
        self._tools[tool.name] = tool

    @beartype
    def get_tool(self, name: str) -> Tool | None:
        """Get a registered tool by name.

        Args:
            name: Tool name to look up

        Returns:
            Tool if found, None otherwise
        """
        return self._tools.get(name)

    @beartype
    def get_all_tools(self) -> list[Tool]:
        """Get all registered tools.

        Returns:
            List of all registered tools
        """
        return list(self._tools.values())

    @beartype
    def get_tool_names(self) -> list[str]:
        """Get list of all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    @beartype
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for all registered tools.

        Returns:
            List of tool schemas in OpenAI function-calling format

        Raises:
            ValueError: If tool has invalid schema configuration
        """
        schemas: list[dict[str, Any]] = []
        for tool in self._tools.values():
            properties: dict[str, Any] = {}
            required_params: list[str] = []

            for param_name, param_def in tool.parameters.items():
                # Validate parameter type
                param_type = param_def.get("type")
                if not param_type:
                    raise ValueError(
                        f"Tool '{tool.name}' parameter '{param_name}' is missing required 'type'"
                    )
                if param_type not in VALID_PARAM_TYPES:
                    raise ValueError(
                        f"Tool '{tool.name}' parameter '{param_name}' has invalid type '{param_type}'. "
                        f"Must be one of: {', '.join(sorted(VALID_PARAM_TYPES))}"
                    )

                prop: dict[str, Any] = {"type": param_type}

                # Copy schema validation keywords
                for key in (
                    "description",
                    "enum",
                    "default",
                    "minimum",
                    "maximum",
                    "minItems",
                    "maxItems",
                    "pattern",
                    "format",
                ):
                    if key in param_def:
                        prop[key] = param_def[key]

                # Handle array item types
                if param_def.get("type") == "array" and "items" in param_def:
                    prop["items"] = param_def["items"]

                # Mark as required if no default
                if "default" not in param_def:
                    required_params.append(param_name)

                properties[param_name] = prop

            schema: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required_params,
                        "additionalProperties": False,
                    },
                },
            }
            schemas.append(schema)
        return schemas

    @beartype
    def get_tools_by_category(self) -> dict[str, list[Tool]]:
        """Group registered tools by category.

        Returns:
            Dictionary mapping category names to lists of tools.
            Tools with no category are grouped under "other".
        """
        categories: dict[str, list[Tool]] = {}
        for tool in self._tools.values():
            category = tool.category or "other"
            if category not in categories:
                categories[category] = []
            categories[category].append(tool)
        return categories

    @beartype
    def get_categories(self) -> set[str]:
        """Get set of all tool categories.

        Returns:
            Set of category names
        """
        return {tool.category or "other" for tool in self._tools.values()}

    @beartype
    def generate_prompt_documentation(
        self,
        group_by_category: bool = True,
        include_categories: set[str] | None = None,
        exclude_categories: set[str] | None = None,
    ) -> str:
        """Generate prompt documentation for all registered tools.

        This creates formatted text suitable for inclusion in system prompts,
        documenting all available tools with their parameters and usage.

        Args:
            group_by_category: If True, group tools by category with headers
            include_categories: If set, only include tools from these categories
            exclude_categories: If set, exclude tools from these categories

        Returns:
            Formatted documentation string for system prompts
        """
        if not self._tools:
            return ""

        sections: list[str] = []

        if group_by_category:
            categories = self.get_tools_by_category()

            # Apply category filters
            if include_categories:
                categories = {k: v for k, v in categories.items() if k in include_categories}
            if exclude_categories:
                categories = {k: v for k, v in categories.items() if k not in exclude_categories}

            # Sort categories for consistent output
            for category in sorted(categories.keys()):
                tools = categories[category]
                if not tools:
                    continue

                # Category header
                category_title = category.replace("_", " ").title()
                sections.append(f"**{category_title} Tools:**\n")

                # Tool documentation
                for tool in sorted(tools, key=lambda t: t.name):
                    sections.append(tool.get_prompt_documentation())
                    sections.append("")  # Blank line between tools
        else:
            # Flat list without categories
            tools = list(self._tools.values())

            # Apply category filters
            if include_categories:
                tools = [t for t in tools if (t.category or "other") in include_categories]
            if exclude_categories:
                tools = [t for t in tools if (t.category or "other") not in exclude_categories]

            for tool in sorted(tools, key=lambda t: t.name):
                sections.append(tool.get_prompt_documentation())
                sections.append("")

        return "\n".join(sections).strip()

    @beartype
    def validate_all_tools(self) -> list[str]:
        """Validate all registered tools.

        Checks:
        - Valid parameter types
        - Required fields present
        - Schema structure validity

        Returns:
            List of validation errors (empty if all valid)
        """
        errors: list[str] = []

        for tool in self._tools.values():
            # Validate parameters
            for param_name, param_def in tool.parameters.items():
                # Check type field exists
                if "type" not in param_def:
                    errors.append(
                        f"Tool '{tool.name}': Parameter '{param_name}' missing 'type' field"
                    )
                    continue

                param_type = param_def["type"]
                if param_type not in VALID_PARAM_TYPES:
                    errors.append(
                        f"Tool '{tool.name}': Parameter '{param_name}' has invalid type "
                        f"'{param_type}'. Valid types: {sorted(VALID_PARAM_TYPES)}"
                    )

        return errors

    @beartype
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name to check

        Returns:
            True if tool is registered
        """
        return name in self._tools

    @beartype
    def count_tools(self) -> int:
        """Get the number of registered tools.

        Returns:
            Number of tools
        """
        return len(self._tools)

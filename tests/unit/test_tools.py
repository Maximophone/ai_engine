import unittest
from typing import Literal
from enum import Enum
from ai_core import tool, Tool, ToolParameter

class TemperatureUnit(Enum):
    CELSIUS = "C"
    FAHRENHEIT = "F"

@tool(
    description="Fetch weather data",
    city="The city name",
    units="The temperature units",
    optional_param="An optional parameter"
)
def get_weather_test(
    city: str,
    units: Literal["celsius", "fahrenheit"] = "celsius",
    optional_param: int = 5
) -> str:
    return f"Weather in {city} is nice in {units}."

@tool(
    description="Fetch weather data using Enum",
    city="The city name",
    units="The temperature units Enum"
)
def get_weather_enum_test(
    city: str,
    units: TemperatureUnit = TemperatureUnit.CELSIUS
) -> str:
    return f"Weather in {city} is nice in {units.name}."

@tool(description="Test Safe Flag", safe=False)
def unsafe_tool_test(param: str) -> str:
    return "unsafe"

@tool(description="Test Safe Flag Default", param="desc")
def safe_tool_test(param: str) -> str:
    return "safe"


class TestToolDecorator(unittest.TestCase):

    def test_tool_creation(self):
        self.assertTrue(hasattr(get_weather_test, 'tool'))
        t = get_weather_test.tool
        self.assertIsInstance(t, Tool)
        self.assertEqual(t.name, "get_weather_test")
        self.assertEqual(t.description, "Fetch weather data")
        self.assertTrue(t.safe) # Default safe=True tested implicitly

    def test_parameter_parsing(self):
        t = get_weather_test.tool
        self.assertIn("city", t.parameters)
        self.assertIn("units", t.parameters)
        self.assertIn("optional_param", t.parameters)

        city_param = t.parameters["city"]
        self.assertEqual(city_param.type, "string")
        self.assertEqual(city_param.description, "The city name")
        self.assertTrue(city_param.required)
        self.assertIsNone(city_param.enum)

        units_param = t.parameters["units"]
        self.assertEqual(units_param.type, "string")
        self.assertEqual(units_param.description, "The temperature units")
        self.assertFalse(units_param.required) # Has default
        self.assertListEqual(units_param.enum, ["celsius", "fahrenheit"])

        optional_param = t.parameters["optional_param"]
        self.assertEqual(optional_param.type, "integer")
        self.assertEqual(optional_param.description, "An optional parameter")
        self.assertFalse(optional_param.required) # Has default
        self.assertIsNone(optional_param.enum)

    def test_enum_parameter_parsing(self):
        t = get_weather_enum_test.tool
        self.assertIn("units", t.parameters)
        units_param = t.parameters["units"]
        self.assertEqual(units_param.type, "string")
        self.assertEqual(units_param.description, "The temperature units Enum")
        self.assertFalse(units_param.required) # Has default
        # Note: Enum members are stored by name
        self.assertListEqual(units_param.enum, ["CELSIUS", "FAHRENHEIT"])

    def test_missing_parameter_description(self):
        with self.assertRaises(ValueError) as cm:
            @tool(description="Tool missing a param description")
            def missing_desc_tool(param1: str, param2: int):
                pass
        self.assertIn("Missing description for parameter 'param1'", str(cm.exception))

    def test_safe_flag(self):
        unsafe_tool = unsafe_tool_test.tool
        safe_tool = safe_tool_test.tool
        default_safe_tool = get_weather_test.tool

        self.assertFalse(unsafe_tool.safe)
        self.assertTrue(safe_tool.safe)
        self.assertTrue(default_safe_tool.safe)

if __name__ == '__main__':
    unittest.main()

# AutoDrafter 系統流程圖

## 1. 主流程（Main Flow）

```mermaid
flowchart TD
    START(["啟動 auto_drafter_system.py"]) --> INIT_LOG["初始化 Logging"]
    INIT_LOG --> FILE_DLG["彈出檔案選擇對話框"]
    FILE_DLG --> SELECT{"選擇 STEP/STP 檔案"}
    SELECT -->|取消| EXIT(["結束"])
    SELECT -->|選擇檔案| INIT_SYS["建立 AutoDraftingSystem"]

    INIT_SYS --> INIT_CAD["建立 MockCADEngine"]
    INIT_SYS --> INIT_AI["建立 AIIntentParser"]

    INIT_CAD --> LOAD_FILE["載入 3D 檔案"]
    INIT_CAD --> LOAD_CFG["載入 drafter_config.json"]

    LOAD_FILE --> EXTRACT["特徵提取 Pipeline"]
    EXTRACT --> INFO_WIN["顯示圖檔資訊視窗"]
    INFO_WIN --> PREVIEW_3D["預覽 3D 模型"]
    PREVIEW_3D --> GEN_DWG["生成 4 張施工圖"]
    GEN_DWG --> PREVIEW_2D["預覽 2D 工程圖"]
    PREVIEW_2D --> DONE(["完成"])

    style START fill:#4CAF50,color:#fff
    style DONE fill:#4CAF50,color:#fff
    style EXIT fill:#f44336,color:#fff
    style GEN_DWG fill:#2196F3,color:#fff
```

## 2. 檔案載入流程（File Loading）

```mermaid
flowchart TD
    INPUT[/"STEP/STP 檔案"/] --> DETECT{"偵測檔案類型"}
    DETECT -->|".step / .stp"| XCAF["XCAF 載入器"]
    DETECT -->|".stl"| STL["PyVista STL 載入"]

    XCAF --> DOC["建立 TDocStd_Document"]
    DOC --> READER["STEPCAFControl_Reader"]
    READER --> READ["ReadFile + Transfer"]
    READ --> SHAPE_TOOL["取得 XCAFDoc_ShapeTool"]
    SHAPE_TOOL --> FREE_SHAPES["GetFreeShapes + Compound"]
    FREE_SHAPES --> CQ_WP["建立 CadQuery Workplane"]

    CQ_WP --> FEATURE_EXT["進入特徵提取 Pipeline"]

    READ -->|"XCAF 失敗"| FALLBACK["cq.importers.importStep"]
    FALLBACK --> FEATURE_EXT

    style INPUT fill:#FF9800,color:#fff
    style FEATURE_EXT fill:#2196F3,color:#fff
```

## 3. 特徵提取 Pipeline（Feature Extraction）

```mermaid
flowchart TD
    FE_START["特徵提取開始"] --> BBOX["計算模型 Bounding Box"]
    BBOX --> SOLIDS{"XCAF 可用?"}
    SOLIDS -->|是| XCAF_EXT["_extract_solids_from_xcaf\n遍歷 XCAF Label 樹"]
    SOLIDS -->|否| RECUR_EXT["遞迴遍歷 Shape 樹\n累積 Transform 矩陣"]

    XCAF_EXT --> SOLID_PROC
    RECUR_EXT --> SOLID_PROC

    SOLID_PROC["處理每個 Solid"] --> TRANSFORM["套用 Transform"]
    TRANSFORM --> CENTROID["BRepGProp - 質心/體積"]
    CENTROID --> SOLID_BBOX["BRepBndLib - Bounding Box"]
    SOLID_BBOX --> STORE_SOLID["儲存至 _solid_shapes"]

    STORE_SOLID --> CIRCLES["提取圓形特徵\nTopExp_Explorer + BRepAdaptor_Curve"]

    CIRCLES --> ADV["進階分析 Pipeline"]

    ADV --> PCL["_extract_pipe_centerlines\n管路中心線提取"]
    ADV --> CLASSIFY["_classify_parts\n零件分類"]
    ADV --> ANGLES["_calculate_angles\n角度計算"]
    ADV --> CUTLIST["_generate_cutting_list\n取料明細"]

    PCL --> DATA_READY[/"特徵資料就緒"/]
    CLASSIFY --> DATA_READY
    ANGLES --> DATA_READY
    CUTLIST --> DATA_READY

    style FE_START fill:#9C27B0,color:#fff
    style DATA_READY fill:#4CAF50,color:#fff
```

## 4. 管路中心線提取（Pipe Centerline Extraction）

```mermaid
flowchart TD
    PCL_START["管路中心線提取"] --> ITER["遍歷 Solid Features"]
    ITER --> METHOD{"分析方法"}

    METHOD -->|"圓柱面"| OCP["OCP Cylinder 分析\nTopExp_Explorer Faces\nBRepAdaptor_Surface"]
    METHOD -->|"BSpline 面"| BSP["BSpline 管路分析\n_analyze_bspline_pipe"]

    OCP --> CYL_AXIS["提取圓柱軸/半徑/中心"]
    CYL_AXIS --> START_END["計算 Start/End 端點"]

    BSP --> SAMPLE["取樣 BSpline 表面路徑"]
    SAMPLE --> FIT_3D["3D 圓擬合修正中心"]
    FIT_3D --> SEG_DETECT["弧/直線段偵測"]

    START_END --> BUILD_PCL
    SEG_DETECT --> BUILD_PCL

    BUILD_PCL["建構 Pipe 資料結構"] --> PCL_OUT[/"pipe_centerlines"/]

    style PCL_START fill:#9C27B0,color:#fff
    style PCL_OUT fill:#4CAF50,color:#fff
```

## 5. 零件分類規則（Part Classification）

```mermaid
flowchart TD
    CLS_START["零件分類"] --> R2{"R2: 支撐架?\n3+ 相同體積 Solid"}
    R2 -->|是| BRACKET["bracket\nconfidence: 0.85"]
    R2 -->|否| R1{"R1: 腳架?\nOCP 管 + slenderness > 3"}
    R1 -->|是| LEG["leg\nconfidence: 0.85"]
    R1 -->|否| R3A{"R3a: 彎軌?\n有 BSpline 面"}
    R3A -->|是| CURVED["track curved\nconfidence: 0.85"]
    R3A -->|否| R3B{"R3b: 直軌?\n管徑/min_dim >= 0.7"}
    R3B -->|是| STRAIGHT["track straight\nconfidence: 0.9"]
    R3B -->|否| R1B{"R1b: 腳架 fallback?\nslenderness > 4"}
    R1B -->|是| LEG_FB["leg fallback\nconfidence: 0.75"]
    R1B -->|否| BASE["base/connector\nconfidence: 0.6"]

    style CLS_START fill:#9C27B0,color:#fff
    style BRACKET fill:#FF9800,color:#fff
    style LEG fill:#FF9800,color:#fff
    style CURVED fill:#FF9800,color:#fff
    style STRAIGHT fill:#FF9800,color:#fff
    style LEG_FB fill:#FF9800,color:#fff
    style BASE fill:#FF9800,color:#fff
```

## 6. 施工圖生成流程（Drawing Generation）

```mermaid
flowchart TD
    GEN_START["generate_sub_assembly_drawing"] --> GET_INFO["get_model_info\n彙總模型資訊"]
    GET_INFO --> STP_DATA["建構 stp_data\n管徑/弧半徑/仰角/弦長..."]
    STP_DATA --> SECTIONS["_detect_track_sections\n軌道分段偵測"]

    SECTIONS --> D0["Drawing 0\n組件總覽圖"]
    SECTIONS --> D1["Drawing 1\n直線段施工圖"]
    SECTIONS --> D2["Drawing 2\n彎軌施工圖"]
    SECTIONS --> D3["Drawing 3\n完整組合施工圖"]

    D0 --> TB0["_build_tb_info + 標題欄"]
    D1 --> TB1["_build_tb_info + 標題欄"]
    D2 --> TB2["_build_tb_info + 標題欄"]
    D3 --> TB3["_build_tb_info + 標題欄"]

    TB0 --> DXF0[/"output/{name}-0.dxf"/]
    TB1 --> DXF1[/"output/{name}-1.dxf"/]
    TB2 --> DXF2[/"output/{name}_2.dxf"/]
    TB3 --> DXF3[/"output/{name}_3.dxf"/]

    style GEN_START fill:#2196F3,color:#fff
    style DXF0 fill:#4CAF50,color:#fff
    style DXF1 fill:#4CAF50,color:#fff
    style DXF2 fill:#4CAF50,color:#fff
    style DXF3 fill:#4CAF50,color:#fff
```

## 7. 各 Drawing 內容與標註旗標

```mermaid
flowchart LR
    subgraph CONFIG["drafter_config.json"]
        F1["show_basic_info"]
        F2["show_leg_angles"]
        F3["show_track_relations"]
    end

    subgraph D0["Drawing 0: 組件總覽圖"]
        D0_ISO["等角視圖 HLR\n方向: 1,1,1"]
        D0_TOP["俯視圖 HLR\n方向: 0,0,-1"]
        D0_DIM["端點跨距 / 弧半徑 R"]
    end

    subgraph D1["Drawing 1: 直線段施工圖"]
        D1_VIEW["側視圖\n上軌 + 下軌 + 腳架"]
        D1_CL["取料明細表"]
        D1_BOM["BOM 表"]
        D1_ANG["腳架夾角標註"]
        D1_REL["軌道關係標註\n中心線距離 / 內側距離 / 末端距離"]
    end

    subgraph D2["Drawing 2: 彎軌施工圖"]
        D2_ARC["正面弧形視圖"]
        D2_ISO["等角立體圖"]
        D2_DIM["R / 弦長 / 仰角 / 高低差"]
    end

    subgraph D3["Drawing 3: 完整組合施工圖"]
        D3_VIEW["完整路徑側視圖"]
        D3_CL["完整取料明細"]
        D3_BOM["完整 BOM"]
        D3_ANG["腳架夾角標註"]
        D3_REL["軌道關係標註"]
    end

    F1 -.->|控制| D0_DIM
    F1 -.->|控制| D1_CL
    F1 -.->|控制| D1_BOM
    F1 -.->|控制| D2_DIM
    F1 -.->|控制| D3_CL
    F1 -.->|控制| D3_BOM

    F2 -.->|控制| D1_ANG
    F2 -.->|控制| D3_ANG

    F3 -.->|控制| D1_REL
    F3 -.->|控制| D3_REL

    style CONFIG fill:#FF9800,color:#fff
```

## 8. 標題欄優先順序（Title Block Priority）

```mermaid
flowchart LR
    OV["1. overrides\n各圖面特定值"] --> CFG["2. drafter_config.json\n使用者設定"]
    CFG --> STP["3. STEP 中繼資料\nFILE_NAME / PRODUCT"]
    STP --> DEF["4. _TB_DEFAULTS\n程式預設值"]

    OV -->|最高優先| RESULT[/"標題欄欄位值"/]
    CFG -->|次優先| RESULT
    STP -->|fallback| RESULT
    DEF -->|最後| RESULT

    style OV fill:#f44336,color:#fff
    style CFG fill:#FF9800,color:#fff
    style STP fill:#2196F3,color:#fff
    style DEF fill:#9E9E9E,color:#fff
    style RESULT fill:#4CAF50,color:#fff
```

## 9. 3D 間距推算流程（Track Gap Calculation）

```mermaid
flowchart TD
    GAP_START["上下軌間距計算"] --> FIND_MAIN["找到上下軌主管段\n有 solid_id 的直線段"]
    FIND_MAIN --> GET_3D["取得 3D 端點 Z 座標\n_compute_path_start_z"]
    GET_3D --> CALC_Y["計算路徑 Y 位移\n_calc_path_y_displacement"]

    CALC_Y --> HAS_3D{"3D 資料可用?"}
    HAS_3D -->|是| START_GAP["start_gap = abs(z_upper - z_lower)"]
    HAS_3D -->|否| FALLBACK["start_gap = spacing / cos(angle)"]

    START_GAP --> END_GAP["end_gap = start_gap + dy_upper - dy_lower"]
    FALLBACK --> END_GAP

    END_GAP --> INNER["內側距離\n= gap * cos(angle) - pipe_dia"]
    END_GAP --> VERT["中心線垂直距離\n= start_gap / end_gap"]
    END_GAP --> END_D["末端距離\n= gap * sin(angle)"]

    style GAP_START fill:#9C27B0,color:#fff
    style INNER fill:#4CAF50,color:#fff
    style VERT fill:#4CAF50,color:#fff
    style END_D fill:#4CAF50,color:#fff
```

## 10. 完整資料流（Data Flow）

```mermaid
flowchart LR
    subgraph INPUT["輸入"]
        STEP[/"STEP 檔案"/]
        CFG[/"drafter_config.json"/]
    end

    subgraph PROCESS["處理 Pipeline"]
        direction TB
        P1["XCAF 載入"] --> P2["Solid 提取"]
        P2 --> P3["管路中心線"]
        P3 --> P4["零件分類"]
        P4 --> P5["角度計算"]
        P5 --> P6["取料明細"]
        P6 --> P7["軌道分段"]
        P7 --> P8["施工圖生成"]
    end

    subgraph OUTPUT["輸出"]
        DXF0[/"{name}-0.dxf\n組件總覽圖"/]
        DXF1[/"{name}-1.dxf\n直線段施工圖"/]
        DXF2[/"{name}_2.dxf\n彎軌施工圖"/]
        DXF3[/"{name}_3.dxf\n完整組合施工圖"/]
    end

    STEP --> P1
    CFG --> P8
    P8 --> DXF0
    P8 --> DXF1
    P8 --> DXF2
    P8 --> DXF3

    style INPUT fill:#FF9800,color:#fff
    style OUTPUT fill:#E8F5E9
```

## 11. 類別關係圖（Class Diagram）

```mermaid
classDiagram
    class AutoDraftingSystem {
        +MockCADEngine cad
        +AIIntentParser parser
        +__init__(model_file)
    }

    class MockCADEngine {
        -Dict _drafter_config
        -List _pipe_centerlines
        -List _part_classifications
        -List _angles
        -Dict _cutting_list
        -Dict _solid_shapes
        +load_3d_file(filepath)
        +get_model_info() Dict
        +generate_sub_assembly_drawing() List
        +generate_overview_drawing()
        -_load_drafter_config(model_file)
        -_build_tb_info(info) Dict
        -_extract_features()
        -_extract_pipe_centerlines() List
        -_classify_parts(features) List
        -_calculate_angles() List
        -_generate_cutting_list() Dict
        -_detect_track_sections() List
        -_compute_path_start_z() float
        -_hlr_project_to_polylines() Tuple
        -_draw_title_block(msp, tb_info)
        -_draw_dimension_line(msp)
    }

    class AIIntentParser {
        +parse(text) Dict
    }

    class GeometricFeature {
        +str id
        +str type
        +dict props
        +str description
    }

    AutoDraftingSystem --> MockCADEngine
    AutoDraftingSystem --> AIIntentParser
    MockCADEngine --> GeometricFeature
```

## 12. 完整呼叫鏈（Call Chain）

```
main()
+-- select_3d_file()
+-- AutoDraftingSystem.__init__()
|   +-- MockCADEngine.__init__()
|   |   +-- load_3d_file()
|   |   |   +-- _load_step_with_xcaf()  [or fallback]
|   |   |   +-- _extract_features()
|   |   |       +-- _extract_solids_from_xcaf()  [or recursive traversal]
|   |   |       +-- _extract_pipe_centerlines()
|   |   |       |   +-- _analyze_bspline_pipe()  [for BSpline]
|   |   |       +-- _classify_parts()
|   |   |       +-- _calculate_angles()
|   |   |       +-- _generate_cutting_list()
|   |   |           +-- _build_track_items()
|   |   +-- _load_drafter_config()
|   +-- AIIntentParser.__init__()
+-- system.cad.get_model_info()
|   +-- _extract_step_metadata()
|   +-- _parse_step_materials()
|   +-- _parse_step_solid_entities()
+-- system.cad.preview_3d_model()
+-- system.cad.generate_sub_assembly_drawing()
    +-- get_model_info()
    +-- _detect_track_sections()
    +-- generate_overview_drawing()         [Drawing 0]
    |   +-- _build_tb_info()
    |   +-- _hlr_project_to_polylines()     [isometric]
    |   +-- _hlr_project_to_polylines()     [top view]
    |   +-- _draw_title_block()
    +-- _draw_straight_section_sheet()      [Drawing 1]
    |   +-- _build_section_cutting_list()
    |   +-- _compute_path_start_z()         [3D gap]
    |   +-- _draw_title_block()
    +-- generate_curved_track_drawing()     [Drawing 2]
    |   +-- _hlr_project_to_polylines()
    |   +-- _draw_title_block()
    +-- generate_section_assembly_drawing() [Drawing 3]
        +-- _draw_rail_path()
        +-- _draw_cutting_list_table()
        +-- _draw_bom_table()
        +-- _draw_title_block()
```

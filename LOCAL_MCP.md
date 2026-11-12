# MCP local 0.2.0

Esta versión habilita el MCP por stdio con almacenamiento SQLite persistente y
búsqueda semántica local. El README histórico también describe funciones de
roadmap; esta guía delimita la implementación comprobada.

## Instalación

Python 3.11 es la versión probada en Windows. Instalar el wheel en un entorno
virtual independiente. `mcp` está limitado a la familia 1.x: la API empleada no
es compatible con 2.x.

```powershell
uv venv --python 3.11 "$env:LOCALAPPDATA\ReflexBrain\runtime"
uv pip install --python "$env:LOCALAPPDATA\ReflexBrain\runtime\Scripts\python.exe" .
codex mcp add reflex-brain --env "REFLEX_DATA_DIR=$env:LOCALAPPDATA\ReflexBrain\data" --env "REFLEX_SCHEDULER_ENABLED=1" --env "PYTHONUTF8=1" -- "$env:LOCALAPPDATA\ReflexBrain\runtime\Scripts\python.exe" -m continual_brain.mcp.server
```

El modelo predeterminado es `all-MiniLM-L6-v2`. Para funcionar sin conexión,
descargarlo previamente y configurar `REFLEX_EMBED_MODEL` con la ruta absoluta
del modelo. En Windows las bibliotecas de aprendizaje automático se importan
antes de abrir stdio, para evitar un bloqueo observado al cargar extensiones
nativas con el lector stdio activo. Los pesos se cargan con la primera consulta
con contenido. Configurar `startup_timeout_sec = 180` y
`tool_timeout_sec = 180` en la sección MCP de Codex para permitir la primera
carga de bibliotecas en este equipo.

## Uso y garantías

- `reflex_remember` guarda contenido explícito y devuelve un ID persistente.
- `reflex_query` consulta contenido aprobado y memorias explícitas. Actualiza
  FAISS al detectar cambios en SQLite, incluidos los rollback. Con
  `include_untrusted=true` incluye contenido web sin aprobar y borradores;
  devuelve su nivel de confianza en los metadatos.
- `reflex_propose_lesson` acepta `topic` y `content`, o mensajes de sesión.
  Devuelve todas las propuestas, persistidas con sus IDs.
- `reflex_apply_refinement` recibe el `proposal_id` en el parámetro histórico
  `refinement_id`. Tras revisar el contenido se usa `approved=true`. Devuelve
  el ID de la aplicación, que se usa para inspeccionarla o revertirla.
- `auto_apply=true` no salta el umbral: solo admite lecciones con peso de
  evidencia de al menos 0.85. El peso sigue siendo heurístico, no una prueba
  de verdad. Las propuestas de contenido explícito y de web exigen revisión.
- La aplicación, snapshot y registro de auditoría se guardan en una sola
  transacción. Repetir una aplicación devuelve el mismo resultado. El rollback
  rechaza sobrescribir cambios posteriores. Las skills se guardan como DRAFT;
  no se ejecuta código generado ni se declara probado sin ejecutarlo.
- `reflex_snapshot` exporta todos los registros de las tres entidades, sin el
  antiguo límite de 1.000. No es un respaldo completo del archivo SQLite.
- `reflex_research` consulta DuckDuckGo HTML sin clave, extrae páginas públicas,
  guarda memorias marcadas como no fiables y propuestas de lecciones. No aplica
  automáticamente las lecciones. La síntesis es una agrupación heurística de
  extractos, no una síntesis mediante un LLM. El proveedor puede bloquear,
  limitar o cambiar su HTML; los fallos se reportan como errores MCP.
- Los accesos web validan IP pública, fijan la IP validada manteniendo Host/TLS,
  revisan cada redirección y limitan la descarga a 2 MB por página. Las páginas
  se tratan como datos, nunca instrucciones ni código ejecutable.
- Las herramientas de scheduler permiten agregar, listar y quitar tareas y
  triggers, comprobar cobertura y consultar estado. Horarios en UTC. Solo se
  ejecutan mientras el proceso MCP propietario está vivo y
  `REFLEX_SCHEDULER_ENABLED=1`; no es un servicio de Windows. No se crean tareas
  ni triggers por defecto. Un lock evita dos ejecutores simultáneos.

## Datos y compatibilidad

Los datos quedan por defecto en `%LOCALAPPDATA%\ReflexBrain\data` en Windows y
`~/.local/share/ReflexBrain/data` en otros sistemas. `REFLEX_DATA_DIR` permite
elegir otra ubicación. No dependen del directorio desde el que Codex se inicia.
SQLite añade `memories.tags`, propuestas y revisión de contenido al abrir bases
0.1. Conviene respaldar las bases existentes antes de migrarlas.

No publicar bases de datos, memorias, snapshots, contenido recopilado, estados
del scheduler, índices, modelos descargados, logs ni exportaciones de uso real.
Estos archivos son locales y están excluidos mediante `.gitignore`. Las pruebas
versionadas emplean ejemplos sintéticos y directorios temporales; no necesitan
memorias ni conversaciones de usuarios.

Los índices de memoria se reconstruyen desde SQLite. El archivo de índice
incluye vectores, IDs y revisión en una publicación atómica; no carga pickle.
Los mapas externos de GraphRAG deben ser JSON. Los antiguos pickle no se leen
automáticamente: requieren una migración supervisada desde una fuente fiable.

El daemon de importación Hermes y sus checkpoints no son necesarios para este
MCP y no se habilitan con esta instalación. Sus problemas pendientes y el
roadmap de aprendizaje continuo no quedan resueltos por esta entrega.

## Validación

Ejecutar `pytest tests` desde un directorio temporal: algunas pruebas históricas
usan rutas relativas. Las regresiones añadidas usan bases e índices aislados,
FAISS real y un encoder determinista para evitar descargas. La prueba de
aceptación stdio se ejecuta además con el wheel instalado y el modelo MiniLM
real, comprobando persistencia entre procesos, aplicar, consultar y revertir.

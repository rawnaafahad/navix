from collections import deque
from dataclasses import dataclass, field

import numpy as np
from nle import nethack

NORTH = 1
EAST = 2
SOUTH = 3
WEST = 4
NORTHEAST = 5
SOUTHEAST = 6
SOUTHWEST = 7
NORTHWEST = 8
ASCEND = 17
DESCEND = 18
WAIT = 19
KICK = 20
SEARCH = 22

DELTA_TO_ACTION = {
    (0, -1): NORTH,
    (1, 0): EAST,
    (0, 1): SOUTH,
    (-1, 0): WEST,
    (1, -1): NORTHEAST,
    (1, 1): SOUTHEAST,
    (-1, 1): SOUTHWEST,
    (-1, -1): NORTHWEST,
}
ACTION_TO_DELTA = {action: delta for delta, action in DELTA_TO_ACTION.items()}
CARDINAL_DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ALL_DELTAS = list(DELTA_TO_ACTION.keys())

# Empirically verified from the installed NLE 1.3.0 diagnostics.
WALL_CMAPS = {0, 1, 2, 3, 4, 5, 6}
WALKABLE_CMAPS = {12, 13, 14, 19, 21, 30}
CLOSED_DOOR_CMAPS = {15, 16}

# Visible map characters that are strong terrain evidence.
VISIBLE_WALKABLE = {".", "#", "<", ">"}
VISIBLE_WALLS = {"-", "|"}

# Glyph classifiers used by the hostile-monster guard. Resolved once at
# import time; if this NLE build does not expose them (unlikely, but
# defensive), the guard becomes a no-op rather than crashing.
_GLYPH_IS_MONSTER = getattr(nethack, "glyph_is_monster", None)
_GLYPH_IS_PET = getattr(nethack, "glyph_is_pet", None)


def player_position(obs):
    blstats = np.asarray(obs["blstats"])
    return int(blstats[0]), int(blstats[1])


def decode_message(obs):
    raw = bytes(np.asarray(obs["message"], dtype=np.uint8).tolist())
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def find_downstairs(obs):
    chars = np.asarray(obs["chars"])
    locations = np.argwhere(chars == ord(">"))
    if len(locations) == 0:
        return None
    row, col = locations[0]
    return int(col), int(row)


def _terrain_cmap(glyph):
    glyph = int(glyph)
    if not nethack.glyph_is_cmap(glyph):
        return None
    return int(nethack.glyph_to_cmap(glyph))


@dataclass
class DungeonMemory:
    walkable: set = field(default_factory=set)
    walls: set = field(default_factory=set)
    closed_doors: set = field(default_factory=set)
    blocked: set = field(default_factory=set)
    known: set = field(default_factory=set)
    downstairs: set = field(default_factory=set)
    visit_counts: dict = field(default_factory=dict)
    searched_counts: dict = field(default_factory=dict)
    exhausted_frontiers: set = field(default_factory=set)

    def update(self, obs):
        chars = np.asarray(obs["chars"])
        glyphs = np.asarray(obs["glyphs"])
        height, width = chars.shape
        player = player_position(obs)

        self.walkable.add(player)
        self.known.add(player)
        self.visit_counts[player] = self.visit_counts.get(player, 0) + 1

        for y in range(height):
            for x in range(width):
                position = (x, y)
                char = chr(int(chars[y, x]))
                glyph = int(glyphs[y, x])

                if char == " ":
                    continue

                # Every visible non-space cell is known. This is important:
                # monsters/items/pets must not masquerade as "unknown" cells
                # and create fake exploration frontiers.
                self.known.add(position)

                # Visible wall geometry overrides stale/dynamic glyph memory.
                if char in VISIBLE_WALLS:
                    self.walls.add(position)
                    self.walkable.discard(position)
                    self.closed_doors.discard(position)
                    self.blocked.discard(position)
                    continue

                # Explicit visible corridor/room/stair terrain overrides glyph
                # ambiguity. In particular '#' must remain traversable.
                if char in VISIBLE_WALKABLE:
                    self.walkable.add(position)
                    self.closed_doors.discard(position)
                    self.walls.discard(position)
                    self.blocked.discard(position)
                    if char == ">":
                        self.downstairs.add(position)
                    continue

                # '+' is the visible closed-door character in the situations
                # relevant to this executor.
                if char == "+":
                    self.closed_doors.add(position)
                    self.walkable.discard(position)
                    self.walls.discard(position)
                    self.blocked.discard(position)
                    continue

                cmap = _terrain_cmap(glyph)

                if cmap is not None:
                    if cmap in CLOSED_DOOR_CMAPS:
                        self.closed_doors.add(position)
                        self.walkable.discard(position)
                        self.walls.discard(position)
                        self.blocked.discard(position)
                    elif cmap in WALKABLE_CMAPS:
                        self.walkable.add(position)
                        self.closed_doors.discard(position)
                        self.walls.discard(position)
                        self.blocked.discard(position)
                    elif cmap in WALL_CMAPS:
                        self.walls.add(position)
                        self.walkable.discard(position)
                        self.closed_doors.discard(position)
                    continue

                # Dynamic entities (pets, monsters, objects, gold, etc.) cover
                # terrain. If this square was previously known as walkable,
                # preserve that knowledge. Crucially, it is already in known,
                # so it cannot create a false frontier merely because a pet or
                # item is visible there.

        self.walkable.add(player)
        self.walls.discard(player)
        self.closed_doors.discard(player)
        self.blocked.discard(player)

    def downstairs_position(self):
        if not self.downstairs:
            return None
        return sorted(self.downstairs)[0]


def _neighbours(position, allow_diagonal=False):
    x, y = position
    deltas = ALL_DELTAS if allow_diagonal else CARDINAL_DELTAS
    for dx, dy in deltas:
        yield x + dx, y + dy


def bfs_path(start, targets, traversable, blocked=None, allow_diagonal=False):
    targets = set(targets)
    if not targets:
        return None
    if start in targets:
        return [start]

    blocked = set() if blocked is None else set(blocked)
    queue = deque([start])
    parent = {start: None}
    found = None

    while queue:
        current = queue.popleft()

        for neighbour in _neighbours(current, allow_diagonal=allow_diagonal):
            if neighbour in parent or neighbour in blocked:
                continue
            if neighbour not in traversable:
                continue

            parent[neighbour] = current

            if neighbour in targets:
                found = neighbour
                queue.clear()
                break

            queue.append(neighbour)

    if found is None:
        return None

    path = []
    current = found
    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path


def bfs_path_cardinal_then_diagonal(start, targets, traversable, blocked=None):
    path = bfs_path(
        start, targets, traversable,
        blocked=blocked, allow_diagonal=False,
    )
    if path is None:
        path = bfs_path(
            start, targets, traversable,
            blocked=blocked, allow_diagonal=True,
        )
    return path


def action_between(current, target):
    delta = (target[0] - current[0], target[1] - current[1])
    if delta not in DELTA_TO_ACTION:
        raise ValueError(f"Unsupported movement delta: {delta}")
    return DELTA_TO_ACTION[delta]


def _step(env, action):
    obs, reward, terminated, truncated, info = env.step(int(action))
    done = bool(terminated or truncated)
    return obs, float(reward), done, info


def _is_yes_no_prompt(obs):
    """Detect NetHack confirmation prompts that can trap the executor."""
    message = decode_message(obs).lower()
    return (
        "[yn]" in message
        or "really attack" in message
        or "still climb?" in message
    )


def _answer_no_if_needed(env, obs, reward, done, info):
    """Resolve a yes/no prompt with 'n' and count the extra primitive step."""
    steps = 1

    if done or not _is_yes_no_prompt(obs):
        return obs, reward, done, info, steps

    resolved_obs, reward_2, done_2, info_2 = _step(env, SOUTHWEST)

    return (
        resolved_obs,
        reward + reward_2,
        done_2,
        info_2,
        steps + 1,
    )


def _kick_direction(env, direction_action):
    obs, reward_1, done, info = _step(env, KICK)

    obs, reward_1, done, info, first_steps = _answer_no_if_needed(
        env, obs, reward_1, done, info
    )

    if done:
        return obs, reward_1, first_steps, done, info

    # If KICK itself produced a prompt, it has already been answered.
    # Do not send the movement direction as another prompt response.
    if first_steps > 1:
        return obs, reward_1, first_steps, done, info

    obs, reward_2, done, info = _step(env, direction_action)

    obs, reward_2, done, info, second_steps = _answer_no_if_needed(
        env, obs, reward_2, done, info
    )

    return (
        obs,
        reward_1 + reward_2,
        first_steps + second_steps,
        done,
        info,
    )

def frontier_positions(memory, map_shape, chars=None):
    """
    Return useful frontiers while suppressing fake corridor-side frontiers.

    The current executor's exhausted-frontier state and visible-wall checks are
    preserved, while the better corridor geometry heuristic from the earlier
    best version is restored.
    """
    height, width = map_shape
    frontiers = set()

    if chars is not None:
        chars = np.asarray(chars)

    def in_bounds(position):
        x, y = position
        return 0 <= x < width and 0 <= y < height

    def unknown(position):
        return in_bounds(position) and position not in memory.known

    for position in memory.walkable:
        if position in memory.exhausted_frontiers:
            continue

        x, y = position
        if not in_bounds(position):
            continue

        if chars is not None and chr(int(chars[y, x])) in VISIBLE_WALLS:
            continue

        if not any(
            unknown((x + dx, y + dy))
            for dx, dy in CARDINAL_DELTAS
        ):
            continue

        north = (x, y - 1) in memory.walkable
        south = (x, y + 1) in memory.walkable
        east = (x + 1, y) in memory.walkable
        west = (x - 1, y) in memory.walkable

        # Interior horizontal corridor: ignore unknown side-space.
        if east and west and not (north or south):
            if unknown((x - 1, y)) or unknown((x + 1, y)):
                frontiers.add(position)
            continue

        # Interior vertical corridor: ignore unknown side-space.
        if north and south and not (east or west):
            if unknown((x, y - 1)) or unknown((x, y + 1)):
                frontiers.add(position)
            continue

        known_walkable_dirs = [
            (dx, dy)
            for dx, dy in CARDINAL_DELTAS
            if (x + dx, y + dy) in memory.walkable
        ]

        # Corridor endpoint: only count unknown continuation directly away
        # from the one known walkable neighbour.
        if len(known_walkable_dirs) == 1:
            dx, dy = known_walkable_dirs[0]
            if unknown((x - dx, y - dy)):
                frontiers.add(position)
            continue

        # Rooms, bends, junctions and door-adjacent floor.
        frontiers.add(position)

    return frontiers

def exit_frontier_positions(memory, map_shape, chars=None):
    """
    Fallback exploration targets for visible room/corridor exits.

    Important: do NOT treat every visible '#' corridor square as an exit.
    Doing so makes the explorer bounce among already-known corridor tiles and
    can consume the whole option budget.

    Prefer only:
      * walkable cells adjacent to genuinely unknown space or a closed door;
      * corridor endpoints/boundaries whose geometry points toward unknown
        continuation.

    Ordinary frontier detection remains the primary mechanism; this is only
    the recovery path when ordinary frontiers are temporarily empty.
    """
    height, width = map_shape
    if chars is not None:
        chars = np.asarray(chars)

    def in_bounds(position):
        x, y = position
        return 0 <= x < width and 0 <= y < height

    def unknown(position):
        return in_bounds(position) and position not in memory.known

    result = set()

    for position in memory.walkable:
        if position in memory.exhausted_frontiers:
            continue

        x, y = position
        if not in_bounds(position):
            continue

        ch = None
        if chars is not None:
            ch = chr(int(chars[y, x]))
            if ch not in {".", "#", "<", ">"}:
                continue

        # A known closed door beside this square is always a meaningful exit.
        if any(
            (x + dx, y + dy) in memory.closed_doors
            for dx, dy in CARDINAL_DELTAS
            if in_bounds((x + dx, y + dy))
        ):
            result.add(position)
            continue

        unknown_dirs = [
            (dx, dy)
            for dx, dy in CARDINAL_DELTAS
            if unknown((x + dx, y + dy))
        ]
        if not unknown_dirs:
            continue

        # For non-corridor floor, adjacency to unknown space is sufficient.
        # This preserves useful room-edge/doorway fallback behaviour.
        if ch != "#":
            result.add(position)
            continue

        # Corridor geometry: only keep a '#' when the unknown space is a
        # plausible continuation of the corridor, not merely unseen side-space.
        walkable_dirs = [
            (dx, dy)
            for dx, dy in CARDINAL_DELTAS
            if (x + dx, y + dy) in memory.walkable
        ]

        # Endpoint: with one known corridor/floor neighbour, continuation is
        # directly away from that neighbour.
        if len(walkable_dirs) == 1:
            dx, dy = walkable_dirs[0]
            if (-dx, -dy) in unknown_dirs:
                result.add(position)
            continue

        north = (0, -1) in walkable_dirs
        south = (0, 1) in walkable_dirs
        east = (1, 0) in walkable_dirs
        west = (-1, 0) in walkable_dirs

        # Straight horizontal corridor: only unknown continuation east/west.
        if east and west and not (north or south):
            if (1, 0) in unknown_dirs or (-1, 0) in unknown_dirs:
                result.add(position)
            continue

        # Straight vertical corridor: only unknown continuation north/south.
        if north and south and not (east or west):
            if (0, -1) in unknown_dirs or (0, 1) in unknown_dirs:
                result.add(position)
            continue

        # Bends/junctions: retain only if unknown space is adjacent. These are
        # relatively rare and can legitimately expose a branch.
        result.add(position)

    return result

def _frontier_score(memory, position):
    visits = memory.visit_counts.get(position, 0)
    searches = memory.searched_counts.get(position, 0)
    return searches, visits


def _adjacent_hostile_monster(obs, position):
    """Detect a non-pet monster glyph in a cardinal cell next to `position`.

    option_explore has no combat logic. Without this guard, a scripted
    SEARCH burst will mechanically run to completion even while an
    adjacent hostile monster attacks the player every turn, which can
    kill the player mid-option (observed with a fox in seed 9's starting
    room). This helper is used only to interrupt SEARCH bursts early; it
    does not attempt to fight or flee. The main loop's ordinary
    frontier/movement/blocked-cell logic takes over on the next outer
    iteration once the burst is interrupted.

    Any non-pet monster is treated as reason to interrupt. An earlier
    version of this function tried to exempt "trivially weak" species
    (e.g. newts) via NLE's monster-level data (permonst().mlevel), to
    stop a wandering newt from indefinitely blocking every SEARCH
    attempt in an enclosed room. That was reverted: NetHack's mlevel
    is a spawn-depth indicator, not a threat/safety score, and several
    genuinely dangerous early monsters -- including the fox that
    originally motivated this guard -- share mlevel 0 with harmless
    ones. The level-based exemption caused that same fox to go
    unnoticed again and kill the player. Do not reintroduce a
    species/level-based exemption without a real threat classification;
    treating any non-pet monster as hostile is the safe default.
    """
    if _GLYPH_IS_MONSTER is None:
        # Defensive fallback if this NLE build doesn't expose the
        # classifier. Do not crash; the guard simply becomes inactive.
        return False

    glyphs = np.asarray(obs["glyphs"])
    height, width = glyphs.shape
    x, y = position

    for dx, dy in CARDINAL_DELTAS:
        nx, ny = x + dx, y + dy
        if not (0 <= ny < height and 0 <= nx < width):
            continue

        glyph = int(glyphs[ny, nx])

        if not _GLYPH_IS_MONSTER(glyph):
            continue

        if _GLYPH_IS_PET is not None and _GLYPH_IS_PET(glyph):
            continue

        return True

    return False


def _wall_adjacent_search_candidates(memory):
    """Walkable cells adjacent to at least one known wall, excluding
    positions already marked exhausted.

    NetHack's SEARCH command checks all 8 surrounding squares (including
    diagonals) for secret doors/passages, not just the 4 cardinal
    neighbours. Using CARDINAL_DELTAS here would miss candidate cells
    whose only adjacent wall segment is diagonal -- observed in seed 7,
    where every option after the room's cardinal-adjacent tiles were
    exhausted returned 0 steps because no further candidates were ever
    offered, even though diagonally-wall-adjacent floor tiles remained.
    ALL_DELTAS matches the real game mechanic.

    Once the current position has been searched
    `search_exhaustion_limit` times with no result, a secret door may
    still exist elsewhere along the room's perimeter -- it can never be
    found by continuing to search from a single fixed spot. This helper
    finds other candidate spots to relocate the search to.
    """
    candidates = set()

    for position in memory.walkable:
        if position in memory.exhausted_frontiers:
            continue

        x, y = position
        if any(
            (x + dx, y + dy) in memory.walls
            for dx, dy in ALL_DELTAS
        ):
            candidates.add(position)

    return candidates


def _move_one_step(env, obs, memory, next_position):
    current = player_position(obs)
    action = action_between(current, next_position)

    if next_position in memory.closed_doors:
        new_obs, reward, steps, done, info = _kick_direction(env, action)

        # Terminal observations in NetHackStaircase are not normal dungeon
        # frames; feeding them into DungeonMemory can create known=1659.
        if not done:
            memory.update(new_obs)

        return new_obs, reward, steps, done

    raw_obs, reward, done, info = _step(env, action)
    movement_prompt = _is_yes_no_prompt(raw_obs)

    new_obs, reward, done, info, steps = _answer_no_if_needed(
        env, raw_obs, reward, done, info
    )

    if done:
        return new_obs, reward, steps, done

    new_position = player_position(new_obs)
    memory.update(new_obs)

    if new_position == current:
        # A pet/peaceful-monster prompt or other failed move should force a
        # replan rather than immediately retrying the same square.
        if next_position not in memory.closed_doors:
            memory.blocked.add(next_position)

        if movement_prompt:
            # Do not permanently exhaust every nearby frontier; the dynamic
            # blocker may move. `blocked` is cleared when normal terrain is
            # visible again in DungeonMemory.update().
            pass
    else:
        memory.blocked.discard(next_position)

    return new_obs, reward, steps, done

def _best_path_to_targets(current, targets, memory):
    traversable = set(memory.walkable) | set(memory.closed_doors)
    best_path = None
    best_score = None
    unreachable = set()

    for target in targets:
        path = bfs_path_cardinal_then_diagonal(
            current,
            [target],
            traversable,
            blocked=memory.blocked,
        )

        if path is None:
            unreachable.add(target)
            continue

        score = (len(path), _frontier_score(memory, target))
        if best_score is None or score < best_score:
            best_score = score
            best_path = path

    return best_path, unreachable


def _stairs_reachable(memory, current):
    """True only if the known '>' tile has a currently-known walkable/door
    path from `current`.

    NetHack can reveal '>' via long-distance line of sight through an open
    room (or occasionally along a long corridor) before the connecting
    tiles between the player and the stairs have actually been seen and
    recorded as walkable. Treating mere visibility (`downstairs_position()
    is not None`) as "ready to navigate" causes option_go_to_downstairs to
    be handed an unreachable target, silently return 0 steps, and report
    failure even though the stairs were genuinely found. This helper is
    the single source of truth for "safe to hand off to navigation".
    """
    target = memory.downstairs_position()
    if target is None:
        return False
    traversable = set(memory.walkable) | set(memory.closed_doors) | {target}
    path = bfs_path_cardinal_then_diagonal(
        current, [target], traversable, blocked=memory.blocked,
    )
    return path is not None


def option_explore(
    env,
    obs,
    memory,
    max_steps=150,
    search_repetitions=5,
    search_exhaustion_limit=20,
):
    total_reward = 0.0
    primitive_steps = 0
    memory.update(obs)

    while primitive_steps < max_steps:
        current = player_position(obs)

        if _stairs_reachable(memory, current):
            return obs, total_reward, primitive_steps, True, False

        chars = np.asarray(obs["chars"])
        frontiers = frontier_positions(memory, chars.shape, chars=chars)

        # PRIORITY 1: approach known closed doors.
        if memory.closed_doors:
            traversable = set(memory.walkable) | set(memory.closed_doors)
            best_door_path = None

            for door in memory.closed_doors:
                path = bfs_path_cardinal_then_diagonal(
                    current,
                    [door],
                    traversable,
                    blocked=memory.blocked,
                )
                if path is None:
                    continue
                if best_door_path is None or len(path) < len(best_door_path):
                    best_door_path = path

            if best_door_path is not None and len(best_door_path) >= 2:
                obs, reward, steps, done = _move_one_step(
                    env, obs, memory, best_door_path[1],
                )
                primitive_steps += steps
                total_reward += reward

                if done:
                    return obs, total_reward, primitive_steps, False, True
                continue

        # Search a genuine frontier, but only a bounded number of times,
        # and never while a hostile monster is standing next to us (see
        # _adjacent_hostile_monster). If a monster is adjacent, skip this
        # branch entirely for now; PRIORITY 2 may move us to a different
        # frontier, or the "nothing reachable" fallback below will detect
        # the same monster and stop the option cleanly instead of
        # standing still and taking free hits.
        if current in frontiers and not _adjacent_hostile_monster(obs, current):
            already_searched = memory.searched_counts.get(current, 0)

            if already_searched < search_repetitions:
                known_before = len(memory.known)
                doors_before = set(memory.closed_doors)
                stairs_before = set(memory.downstairs)

                new_obs, reward, done, info = _step(env, SEARCH)
                primitive_steps += 1
                total_reward += reward
                memory.searched_counts[current] = already_searched + 1
                obs = new_obs

                if done:
                    return obs, total_reward, primitive_steps, False, True

                memory.update(obs)

                if _stairs_reachable(memory, player_position(obs)):
                    return obs, total_reward, primitive_steps, True, False

                progressed = (
                    len(memory.known) > known_before
                    or set(memory.closed_doors) != doors_before
                    or set(memory.downstairs) != stairs_before
                )

                if (
                    not progressed
                    and memory.searched_counts.get(current, 0)
                    >= search_repetitions
                ):
                    memory.exhausted_frontiers.add(current)

                continue

            memory.exhausted_frontiers.add(current)
            frontiers.discard(current)

        # PRIORITY 2: ordinary unknown-neighbour frontiers.
        candidate_frontiers = {
            position for position in frontiers if position != current
        }

        if candidate_frontiers:
            best_path, unreachable = _best_path_to_targets(
                current, candidate_frontiers, memory,
            )
            memory.exhausted_frontiers.update(unreachable)

            if best_path is not None and len(best_path) >= 2:
                target = best_path[-1]
                before_position = current
                known_before = len(memory.known)

                obs, reward, steps, done = _move_one_step(
                    env, obs, memory, best_path[1],
                )
                primitive_steps += steps
                total_reward += reward

                if done:
                    return obs, total_reward, primitive_steps, False, True

                after_position = player_position(obs)
                if (
                    after_position == before_position
                    and len(memory.known) == known_before
                ):
                    memory.exhausted_frontiers.add(target)

                continue

        # PRIORITY 3: visible corridor/exit fallback.
        #
        # This fixes the failure mode where frontier_positions() becomes empty
        # even though a visible '#' corridor leads out of the room. We move
        # toward the least-visited reachable corridor/exit before resorting to
        # blind SEARCH.
        chars = np.asarray(obs["chars"])
        fallback_targets = exit_frontier_positions(
            memory, chars.shape, chars=chars,
        )
        fallback_targets.discard(current)

        if fallback_targets:
            traversable = set(memory.walkable) | set(memory.closed_doors)
            best_path = None
            best_score = None
            unreachable = set()

            for target in fallback_targets:
                path = bfs_path_cardinal_then_diagonal(
                    current,
                    [target],
                    traversable,
                    blocked=memory.blocked,
                )

                if path is None:
                    unreachable.add(target)
                    continue

                score = (
                    memory.visit_counts.get(target, 0),
                    memory.searched_counts.get(target, 0),
                    len(path),
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best_path = path

            memory.exhausted_frontiers.update(unreachable)

            if best_path is not None and len(best_path) >= 2:
                target = best_path[-1]
                before_position = current
                known_before = len(memory.known)

                obs, reward, steps, done = _move_one_step(
                    env, obs, memory, best_path[1],
                )
                primitive_steps += steps
                total_reward += reward

                if done:
                    return obs, total_reward, primitive_steps, False, True

                after_position = player_position(obs)
                if (
                    after_position == before_position
                    and len(memory.known) == known_before
                ):
                    memory.exhausted_frontiers.add(target)

                continue

        # Nothing reachable via doors/frontiers/exits remains. Repeatedly
        # SEARCH this position in bounded bursts until one of:
        #   - progress is made (new known cells / doors / stairs revealed
        #     -> break out to let the main loop reassess priorities), or
        #   - a hostile monster becomes adjacent (stop rather than absorb
        #     free hits while standing still), or
        #   - this position hits `search_exhaustion_limit` total searches,
        #     in which case we try relocating to another wall-adjacent
        #     unexhausted position rather than giving up outright (a
        #     secret door may exist elsewhere along the room's
        #     perimeter -- NetHack only checks cells adjacent to the
        #     player's CURRENT square), or
        #   - the option's step budget runs out.
        #
        # A single burst of `search_repetitions` (5) is not reliably
        # enough to reveal a NetHack secret door, which can require many
        # more attempts. Giving up after exactly one burst was causing
        # options to terminate at a small fraction of their step budget
        # (e.g. 5 steps out of 150) even in seeds where the only way
        # forward was a secret door in the current room.
        current = player_position(obs)
        stuck = False

        while True:
            monster_adjacent = _adjacent_hostile_monster(obs, current)
            already_searched = memory.searched_counts.get(current, 0)
            exhausted = already_searched >= search_exhaustion_limit

            if monster_adjacent or exhausted:
                if exhausted:
                    memory.exhausted_frontiers.add(current)

                # Either a hostile monster is blocking search from this
                # position, or this position's search budget is spent.
                # In both cases, try relocating to a different
                # wall-adjacent, unexhausted cell before giving up on
                # the option entirely -- reuses the same BFS/scoring
                # machinery used for ordinary frontier movement
                # (PRIORITY 2) rather than inventing new logic. Without
                # this, a monster wandering next to the player at a
                # dead end causes the option to give up in 0 steps even
                # when other unexplored perimeter remains reachable.
                relocated = False
                move_failed = False
                candidates = (
                    _wall_adjacent_search_candidates(memory) - {current}
                )

                if monster_adjacent:
                    # Prefer candidates that are not, as far as the
                    # current snapshot shows, already adjacent to the
                    # same monster. Best-effort only -- the monster may
                    # move -- so fall back to the unfiltered set if
                    # every candidate currently looks monster-adjacent.
                    non_adjacent_candidates = {
                        c for c in candidates
                        if not _adjacent_hostile_monster(obs, c)
                    }
                    if non_adjacent_candidates:
                        candidates = non_adjacent_candidates

                if candidates and primitive_steps < max_steps:
                    reloc_path, unreachable = _best_path_to_targets(
                        current, candidates, memory,
                    )
                    memory.exhausted_frontiers.update(unreachable)

                    if reloc_path is not None and len(reloc_path) >= 2:
                        move_failed = False

                        for step_index in range(1, len(reloc_path)):
                            if primitive_steps >= max_steps:
                                break

                            before_position = player_position(obs)

                            # Defensive: the precomputed path assumes the
                            # player is standing on the previous path
                            # node. Monsters, pets, or other side effects
                            # (e.g. _answer_no_if_needed sending a real
                            # movement action to dismiss a yes/no prompt)
                            # can move the player somewhere the path
                            # never anticipated. Feeding a stale path
                            # node into _move_one_step then crashes
                            # action_between with a huge, non-adjacent
                            # delta (observed during real training after
                            # ~800 option calls). If reality no longer
                            # matches the plan, abandon this relocation
                            # attempt safely instead of crashing -- the
                            # existing move_failed handling below already
                            # retries from wherever the player actually
                            # is.
                            if before_position != reloc_path[step_index - 1]:
                                move_failed = True
                                break

                            obs, reward, steps, done = _move_one_step(
                                env, obs, memory, reloc_path[step_index],
                            )
                            primitive_steps += steps
                            total_reward += reward

                            if done:
                                return (
                                    obs, total_reward, primitive_steps,
                                    False, True,
                                )

                            if _stairs_reachable(
                                memory, player_position(obs),
                            ):
                                return (
                                    obs, total_reward, primitive_steps,
                                    True, False,
                                )

                            if player_position(obs) == before_position:
                                # Movement failed partway (newly blocked
                                # cell, dynamic obstruction). Abandon this
                                # relocation attempt rather than looping
                                # on a path that no longer works.
                                move_failed = True
                                break

                        if (
                            not move_failed
                            and player_position(obs) in candidates
                        ):
                            relocated = True
                            current = player_position(obs)

                if relocated:
                    # Resume the inner burst loop from the new position.
                    continue

                if move_failed:
                    # The relocation path was blocked partway through by
                    # a dynamic obstruction (e.g. a monster wandered
                    # into the route -- moving into an occupied square
                    # attacks instead of moving, per NetHack semantics,
                    # observed with a lichen in seed 7). _move_one_step
                    # already recorded the blocked cell in memory.blocked,
                    # so retry relocation from the current position
                    # rather than giving up outright: BFS will route
                    # around the newly-known obstruction or fall back to
                    # a different candidate on the next attempt.
                    current = player_position(obs)
                    continue

                stuck = True
                break

            remaining_budget = max_steps - primitive_steps
            burst = min(
                search_repetitions,
                search_exhaustion_limit - already_searched,
                remaining_budget,
            )

            if burst <= 0:
                stuck = True
                break

            known_before = len(memory.known)
            doors_before = set(memory.closed_doors)
            stairs_before = set(memory.downstairs)
            monster_interrupt = False

            for _ in range(burst):
                new_obs, reward, done, info = _step(env, SEARCH)
                primitive_steps += 1
                total_reward += reward
                memory.searched_counts[current] = (
                    memory.searched_counts.get(current, 0) + 1
                )
                obs = new_obs

                if done:
                    return obs, total_reward, primitive_steps, False, True

                memory.update(obs)

                if _stairs_reachable(memory, player_position(obs)):
                    return obs, total_reward, primitive_steps, True, False

                if _adjacent_hostile_monster(obs, player_position(obs)):
                    monster_interrupt = True
                    break

            new_chars = np.asarray(obs["chars"])
            new_frontiers = frontier_positions(
                memory, new_chars.shape, chars=new_chars,
            )
            new_exits = exit_frontier_positions(
                memory, new_chars.shape, chars=new_chars,
            )

            progressed = (
                len(memory.known) > known_before
                or set(memory.closed_doors) != doors_before
                or set(memory.downstairs) != stairs_before
            )

            if progressed or new_frontiers or (new_exits - {current}):
                # New information revealed. Break out of the burst loop
                # so the OUTER while loop re-evaluates doors/frontiers/
                # exits from scratch rather than blindly continuing to
                # search the same square.
                break

            if monster_interrupt:
                # Do not give up immediately: `continue` re-enters the
                # top of this inner loop, where `monster_adjacent` is
                # re-evaluated fresh and, if still non-trivially
                # hostile, triggers the same relocation attempt used
                # when a monster is detected at the start of an
                # iteration (see the `monster_adjacent or exhausted`
                # branch above). Without this, a monster interrupting
                # mid-burst gave up on the whole option immediately,
                # inconsistently with the top-of-loop case.
                continue

            # No progress yet and not exhausted: loop again for another
            # burst at the same position.

        if stuck:
            break

    success = _stairs_reachable(memory, player_position(obs))
    return obs, total_reward, primitive_steps, success, False


def option_go_to_downstairs(env, obs, memory, max_steps=500):
    total_reward = 0.0
    primitive_steps = 0
    memory.update(obs)

    target = memory.downstairs_position()

    if target is None:
        return obs, total_reward, primitive_steps, False, False

    while primitive_steps < max_steps:
        current = player_position(obs)

        if current == target:
            return obs, total_reward, primitive_steps, True, False

        traversable = (
            set(memory.walkable)
            | set(memory.closed_doors)
            | {target}
        )

        # Cardinal routes are preferred, but do not fail navigation merely
        # because the known route requires a diagonal NetHack step.
        path = bfs_path_cardinal_then_diagonal(
            current,
            [target],
            traversable,
            blocked=memory.blocked,
        )

        if path is None or len(path) < 2:
            return obs, total_reward, primitive_steps, False, False

        obs, reward, steps, done = _move_one_step(
            env, obs, memory, path[1],
        )

        primitive_steps += steps
        total_reward += reward

        if done:
            # NetHackStaircase terminates when the staircase objective is
            # reached. A positive transition reward therefore means success.
            success = reward > 0.0
            return obs, total_reward, primitive_steps, success, True

    return obs, total_reward, primitive_steps, False, False


def option_descend(env, obs, memory):
    memory.update(obs)

    current = player_position(obs)
    target = memory.downstairs_position()

    if target is None or current != target:
        return obs, 0.0, 0, False, False

    new_obs, reward, done, info = _step(env, DESCEND)

    if not done:
        memory.update(new_obs)

    # Generic termination is not automatically success.
    success = reward > 0.0

    return new_obs, reward, 1, success, done
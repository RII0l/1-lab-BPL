from pathlib import Path
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

# Поле клеточного автомата.
# Каждая клетка хранит целое число:
# 0 - пусто
# 1 - автомобиль вправо
# 2 - автомобиль вниз
Field = npt.NDArray[np.int8]
FloatArray = npt.NDArray[np.float64]

# Одна цветовая схема используется во всей программе:
# 0 - белый      -> пустая клетка
# 1 - синий      -> автомобиль вправо
# 2 - красный    -> автомобиль вниз
BML_CMAP = ListedColormap(
    [
        "white",
        "royalblue",
        "crimson"
    ]
)

def create_field(
    size: int,
    density: float,
    east_fraction: float,
    seed: int | None = None
) -> Field:
    """
    Создает начальное состояние двумерной BML-модели.

    Параметры
    ----------
    size:
        Размер квадратного поля size x size.

    density:
        Общая плотность автомобилей.
        Допустимый диапазон: 0 <= density <= 1.

    east_fraction:
        Доля автомобилей, движущихся вправо.
        Остальные автомобили движутся вниз.

    seed:
        Зерно генератора случайных чисел.
        Позволяет воспроизводить одинаковое
        начальное состояние модели.

    Возвращает
    ----------
    field:
        Двумерный массив состояний клеток.
    """

    if size <= 0:
        raise ValueError(
            "Размер поля должен быть положительным."
        )
    if not 0.0 <= density <= 1.0:
        raise ValueError(
            "Плотность должна находиться в диапазоне от 0 до 1."
        )
    if not 0.0 <= east_fraction <= 1.0:
        raise ValueError(
            "Доля машин, движущихся вправо, "
            "должна находиться в диапазоне от 0 до 1."
        )
    
    rng = np.random.default_rng(seed)
    field: Field = np.zeros(
        (size, size),
        dtype=np.int8
    )
    number_of_cells = size * size
    number_of_cars = int(
        density * number_of_cells
    )
    number_east = int(
        number_of_cars * east_fraction
    )
    positions = rng.choice(
        number_of_cells,
        size=number_of_cars,
        replace=False
    )
    east_positions = positions[
        :number_east
    ]
    south_positions = positions[
        number_east:
    ]
    east_rows, east_cols = np.unravel_index(
        east_positions,
        (size, size)
    )
    south_rows, south_cols = np.unravel_index(
        south_positions,
        (size, size)
    )
    field[
        east_rows,
        east_cols
    ] = 1
    field[
        south_rows,
        south_cols
    ] = 2
    return field

def bml_step(
    field: Field
) -> tuple[Field, int]:
    east_cars = (
        field == 1
    )
    right_cells_empty = (
        np.roll(
            field,
            shift=-1,
            axis=1
        ) == 0
    )
    east_can_move = (
        east_cars
        & right_cells_empty
    )
    moved_east = int(
        np.sum(
            east_can_move
        )
    )
    field_after_east: Field = (
        field.copy()
    )
    field_after_east[
        east_can_move
    ] = 0
    east_new_positions = np.roll(
        east_can_move,
        shift=1,
        axis=1
    )
    field_after_east[
        east_new_positions
    ] = 1
    south_cars = (
        field_after_east == 2
    )
    lower_cells_empty = (
        np.roll(
            field_after_east,
            shift=-1,
            axis=0
        ) == 0
    )
    south_can_move = (
        south_cars
        & lower_cells_empty
    )
    moved_south = int(
        np.sum(
            south_can_move
        )
    )
    new_field: Field = (
        field_after_east.copy()
    )
    new_field[
        south_can_move
    ] = 0
    south_new_positions = np.roll(
        south_can_move,
        shift=1,
        axis=0
    )
    new_field[
        south_new_positions
    ] = 2
    moved = (
        moved_east
        + moved_south
    )
    return new_field, moved

def run_simulation(
    size: int,
    density: float,
    steps: int,
    east_fraction: float,
    seed: int | None = None
) -> tuple[Field, FloatArray]:
    if steps <= 0:
        raise ValueError(
            "Количество тактов должно быть положительным."
        )
    field = create_field(
        size=size,
        density=density,
        east_fraction=east_fraction,
        seed=seed
    )
    number_of_cars = int(
        np.count_nonzero(
            field
        )
    )
    velocities_list: list[float] = []

    for _ in range(steps):
        field, moved = bml_step(
            field
        )
        if number_of_cars > 0:
            velocity = (
                moved
                / number_of_cars
            )

        else:
            velocity = 0.0
        velocities_list.append(
            velocity
        )
    velocities: FloatArray = np.asarray(
        velocities_list,
        dtype=np.float64
    )
    return field, velocities

def density_experiment(
    size: int,
    steps: int,
    east_fraction: float,
    transient: int,
    repetitions: int,
    density_min: float,
    density_max: float,
    density_step: float
) -> tuple[FloatArray, FloatArray]:
    if transient < 0:
        raise ValueError(
            "transient не может быть отрицательным."
        )
    if transient >= steps:
        raise ValueError(
            "transient должен быть меньше steps."
        )
    if repetitions <= 0:
        raise ValueError(
            "Количество повторений должно быть положительным."
        )
    if density_step <= 0:
        raise ValueError(
            "Шаг плотности должен быть положительным."
        )
    
    densities: FloatArray = np.arange(
        density_min,
        density_max + density_step / 2,
        density_step,
        dtype=np.float64
    )
    mean_velocities_list: list[float] = []
    for density in densities:
        experiment_results: list[float] = []
        for repetition in range(
            repetitions
        ):
            _, velocities = run_simulation(
                size=size,
                density=float(density),
                steps=steps,
                east_fraction=east_fraction,
                seed=repetition
            )
            stationary_velocity = float(
                np.mean(
                    velocities[
                        transient:
                    ]
                )
            )
            experiment_results.append(
                stationary_velocity
            )
        mean_velocity = float(
            np.mean(
                experiment_results
            )
        )
        mean_velocities_list.append(
            mean_velocity
        )
        print(
            f"Плотность = {density:.2f}, "
            f"средняя скорость = {mean_velocity:.3f}"
        )
    mean_velocities: FloatArray = np.asarray(
        mean_velocities_list,
        dtype=np.float64
    )
    return (
        densities,
        mean_velocities
    )

def show_field(
    field: Field,
    title: str
) -> None:
    plt.figure(
        figsize=(7, 7)
    )
    plt.imshow(
        field,
        cmap=BML_CMAP,
        vmin=0,
        vmax=2,
        interpolation="nearest"
    )
    plt.title(
        title
    )
    plt.xlabel(
        "x"
    )
    plt.ylabel(
        "y"
    )
    plt.grid(
        False
    )
    plt.tight_layout()
    plt.show()

def animate_simulation(
    size: int,
    density: float,
    steps: int,
    interval: int,
    east_fraction: float,
    seed: int | None,
    save_snapshots: bool,
    snapshot_step: int
) -> FuncAnimation:

    if steps <= 0:
        raise ValueError(
            "Количество тактов должно быть положительным."
        )
    if interval <= 0:
        raise ValueError(
            "Интервал должен быть положительным."
        )
    if snapshot_step <= 0:
        raise ValueError(
            "Шаг сохранения должен быть положительным."
        )

    field = create_field(
        size=size,
        density=density,
        east_fraction=east_fraction,
        seed=seed
    )

    number_of_cars = int(
        np.count_nonzero(
            field
        )
    )
    states: list[Field] = [
        field.copy()
    ]
    velocities: list[float] = [
        0.0
    ]

    for _ in range(steps):
        field, moved = bml_step(
            field
        )
        if number_of_cars > 0:
            velocity = (
                moved
                / number_of_cars
            )
        else:
            velocity = 0.0
        states.append(
            field.copy()
        )
        velocities.append(
            velocity
        )
    save_path = (
        Path(__file__)
        .resolve()
        .parent
    )

    if save_snapshots:
        for step_number in range(
            snapshot_step,
            steps + 1,
            snapshot_step
        ):
            filename = (
                save_path
                / f"state_t_{step_number:04d}.png"
            )
            fig_snapshot, ax_snapshot = plt.subplots(
                figsize=(7, 7)
            )
            ax_snapshot.imshow(
                states[step_number],
                cmap=BML_CMAP,
                vmin=0,
                vmax=2,
                interpolation="nearest"
            )
            ax_snapshot.set_title(
                f"BML-модель | "
                f"ρ = {density:.2f} | "
                f"t = {step_number} | "
                f"<v> = {velocities[step_number]:.3f}"
            )
            ax_snapshot.set_xlabel(
                "x"
            )
            ax_snapshot.set_ylabel(
                "y"
            )
            ax_snapshot.grid(
                False
            )
            fig_snapshot.tight_layout()
            fig_snapshot.savefig(
                filename,
                dpi=200,
                bbox_inches="tight"
            )
            plt.close(
                fig_snapshot
            )
            print(
                f"Сохранено изображение: "
                f"{filename}"
            )
    fig, ax = plt.subplots(
        figsize=(7, 7)
    )
    image = ax.imshow(
        states[0],
        cmap=BML_CMAP,
        vmin=0,
        vmax=2,
        interpolation="nearest"
    )
    ax.set_xlabel(
        "x"
    )
    ax.set_ylabel(
        "y"
    )
    ax.grid(
        False
    )
    title = ax.set_title(
        f"BML-модель | "
        f"ρ = {density:.2f} | "
        f"t = 0"
    )
    plt.tight_layout()

  
    def update(
        frame_number: int
    ) -> tuple:
        image.set_data(
            states[frame_number]
        )
        title.set_text(
            f"BML-модель | "
            f"ρ = {density:.2f} | "
            f"t = {frame_number} | "
            f"<v> = {velocities[frame_number]:.3f}"
        )
        return (
            image,
            title
        )
    
    animation = FuncAnimation(
        fig,
        update,
        frames=steps + 1,
        interval=interval,
        repeat=False,
        blit=False
    )
    plt.show()
    return animation

def plot_velocity_time(
    velocities: FloatArray,
    density: float
) -> None:
    time_steps = np.arange(
        1,
        len(velocities) + 1
    )
    plt.figure(
        figsize=(9, 5)
    )
    plt.plot(
        time_steps,
        velocities
    )
    plt.xlabel(
        "Номер такта t"
    )
    plt.ylabel(
        "Средняя скорость <v>"
    )
    plt.title(
        f"Изменение средней скорости "
        f"при ρ = {density:.2f}"
    )
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_velocity_density(
    densities: FloatArray,
    mean_velocities: FloatArray
) -> None:
    plt.figure(
        figsize=(9, 5)
    )
    plt.plot(
        densities,
        mean_velocities,
        "o-"
    )
    plt.xlabel(
        "Плотность автомобилей ρ"
    )
    plt.ylabel(
        "Средняя скорость <v>"
    )
    plt.title(
        "Зависимость средней скорости "
        "от плотности автомобилей"
    )
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_flow_density(
    densities: FloatArray,
    mean_velocities: FloatArray
) -> None:
    flow: FloatArray = (
        densities
        * mean_velocities
    )
    plt.figure(
        figsize=(9, 5)
    )
    plt.plot(
        densities,
        flow,
        "o-"
    )
    plt.xlabel(
        "Плотность автомобилей ρ"
    )
    plt.ylabel(
        "Транспортный поток j"
    )
    plt.title(
        "Зависимость транспортного потока "
        "от плотности автомобилей"
    )
    plt.grid()
    plt.tight_layout()
    plt.show()

def main() -> None:


    # ========================================================
    # ПОЛЬЗОВАТЕЛЬСКИЕ НАСТРОЙКИ
    # ========================================================

    # --------------------------------------------------------
    # РЕЖИМ РАБОТЫ
    # --------------------------------------------------------
    #
    # "animation"  - показать движение автомобилей
    #
    # "single"     - выполнить один расчет,
    #                построить <v>(t)
    #                и показать конечное состояние
    #
    # "experiment" - исследовать влияние плотности,
    #                построить <v>(rho) и j(rho)
    #
    mode: str = "animation"


    # --------------------------------------------------------
    # ОБЩИЕ ПАРАМЕТРЫ МОДЕЛИ
    # --------------------------------------------------------

    # size = 50 -> поле 50 x 50
    size: int = 100
    density: float = 0.4 # используется в режимах animation и single
    steps: int = 300
    east_fraction: float = 0.5
    seed: int = 1


    # --------------------------------------------------------
    # ПАРАМЕТРЫ АНИМАЦИИ
    # --------------------------------------------------------

    animation_interval: int = 100
    save_snapshots: bool = True
    snapshot_step: int = 50


    transient: int = 200
    repetitions: int = 1
    density_min: float = 0.05
    density_max: float = 0.95
    density_step: float = 0.05


    if mode == "animation":

        animate_simulation(
            size=size,
            density=density,
            steps=steps,
            interval=animation_interval,
            east_fraction=east_fraction,
            seed=seed,
            save_snapshots=save_snapshots,
            snapshot_step=snapshot_step
        )

    elif mode == "single":

        final_field, velocities = run_simulation(
            size=size,
            density=density,
            steps=steps,
            east_fraction=east_fraction,
            seed=seed
        )
        plot_velocity_time(
            velocities,
            density
        )
        show_field(
            final_field,
            title=(
                f"Конечное состояние системы | "
                f"ρ = {density:.2f} | "
                f"t = {steps}"
            )
        )

    elif mode == "experiment":

        densities, mean_velocities = density_experiment(
            size=size,
            steps=steps,
            east_fraction=east_fraction,
            transient=transient,
            repetitions=repetitions,
            density_min=density_min,
            density_max=density_max,
            density_step=density_step
        )


        # средняя скорость от плотности
        plot_velocity_density(
            densities,
            mean_velocities
        )

    
        plot_flow_density(
            densities,
            mean_velocities
        )


    
    else:

        raise ValueError(
            "Неизвестный режим работы. "
            "Используйте 'animation', "
            "'single' или 'experiment'."
        )


if __name__ == "__main__":
    main()